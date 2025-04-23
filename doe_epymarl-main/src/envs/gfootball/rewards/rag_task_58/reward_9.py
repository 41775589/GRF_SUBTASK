import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward function to emphasize defensive strategies
    and ball distribution over simply scoring goals.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ownership_change_reward = 0.1
        self.ball_recovery_reward = 0.2
        self.defensive_position_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "possession_change_reward": [0.0] * len(reward),
            "ball_recovery_reward": [0.0] * len(reward),
            "defensive_position_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' not in o:
                continue

            prev_ball_owned_team = o.get('prev_ball_owned_team', -1)
            current_ball_owned_team = o['ball_owned_team']

            # Reward change in possession to our team
            if current_ball_owned_team == o['active'] and prev_ball_owned_team != o['active']:
                reward[rew_index] += self.ownership_change_reward
                components["possession_change_reward"][rew_index] = self.ownership_change_reward

            # Reward for ball recovery by defensive action
            if o['active'] in o['left_team_tired_factor'] and o['ball_owned_team'] == 0:
                tired_factor = o['left_team_tired_factor'][o['active']]
                reward[rew_index] += tired_factor * self.ball_recovery_reward
                components["ball_recovery_reward"][rew_index] = tired_factor * self.ball_recovery_reward

            # Reward for maintaining a good defensive position
            our_team_pos = o['left_team'] if o['active'] in o['left_team_online'] else o['right_team']
            ball_pos = o['ball'][0:2]
            positions = our_team_pos[:, :-1]

            # Calculate if the agent is between the ball and the own goal
            goal_pos = [-1, 0] if o['active'] in o['left_team_online'] else [1, 0]
            are_between = np.logical_and(
                (positions[:, 0] > ball_pos[0]) == (goal_pos[0] > ball_pos[0]), 
                (positions[:, 1] > ball_pos[1]) == (goal_pos[1] > ball_pos[1])
            )

            reward[rew_index] += np.sum(are_between) * self.defensive_position_reward
            components["defensive_position_reward"][rew_index] = np.sum(are_between) * self.defensive_position_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
