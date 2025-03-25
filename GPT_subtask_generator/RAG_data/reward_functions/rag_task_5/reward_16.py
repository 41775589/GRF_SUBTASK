import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive behavior specialized reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.ball_possession_changes = 0
        self.last_ball_owned_team = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.transition_reward = 0.05
        self.defensive_position_reward = 0.1

    def reset(self):
        self.ball_possession_changes = 0
        self.last_ball_owned_team = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "transition_reward": [0.0] * len(reward),
            "defensive_position_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            current_ball_owned_team = o.get('ball_owned_team')

            if current_ball_owned_team != self.last_ball_owned_team:
                if self.last_ball_owned_team is not None:
                    # Reward for changing possession
                    self.ball_possession_changes += 1
                    components["transition_reward"][rew_index] = self.transition_reward
                    reward[rew_index] += components["transition_reward"][rew_index]

            # Encourage defensive position near their own goal
            if current_ball_owned_team == 1 and o['left_team_roles'][o['active']] != 0:
                player_pos = o['left_team'][o['active']]
                goal_pos = [-1, 0] # Typically the left team's goal
                dist_to_goal = np.linalg.norm(np.array(player_pos) - np.array(goal_pos))
                if dist_to_goal < 0.2: # Close to goal
                    components["defensive_position_reward"][rew_index] = self.defensive_position_reward
                    reward[rew_index] += components["defensive_position_reward"][rew_index]

            self.last_ball_owned_team = current_ball_owned_team

        return reward, components

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        rew, components = self.reward(rew)
        info["final_reward"] = sum(rew)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return obs, rew, done, info
