import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward based on refined shooting and dribbling in offensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribbling_reward_multiplier = 1.2
        self.shooting_reward_multiplier = 1.8
        self.goal_distance_threshold = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "dribbling_reward": [0.0], "shooting_reward": [0.0]}

        if observation is None:
            return reward, components

        o = observation[0]
        if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
            ball_position = np.array(o['ball'][:2])
            goal_position = np.array([1.0, 0])  # Assumption of shooting towards the right goal
            distance_to_goal = np.linalg.norm(ball_position - goal_position)

            # Dribbling reward for controlled dribble towards goal
            if o['sticky_actions'][9] == 1:
                components["dribbling_reward"][0] = self.dribbling_reward_multiplier
                reward[0] += components["dribbling_reward"][0]
            
            # Shooting reward for actions taken in proximity to goal
            if distance_to_goal < self.goal_distance_threshold and (o['sticky_actions'][5] == 1 or o['sticky_actions'][6] == 1):
                components["shooting_reward"][0] = self.shooting_reward_multiplier
                reward[0] += components["shooting_reward"][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        self.sticky_actions_counter.fill(0)
        obs = observation if isinstance(observation, list) else [observation]
        for agent_obs in obs:
            if 'sticky_actions' in agent_obs:
                for i, action_state in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = action_state
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
