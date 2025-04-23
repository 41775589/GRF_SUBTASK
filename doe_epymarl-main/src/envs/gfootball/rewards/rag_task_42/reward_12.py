import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for mastering midfield dynamics including enhanced coordination under pressure 
    and strategic repositioning for offensive and defensive transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.midfield_position = 0.0  # Midfield x-coordinate
        self.pressure_threshold = 0.2  # Threshold for 'under pressure'
        self.repositioning_reward = 0.1  # Reward for effective repositioning
        self.coordination_reward = 0.1  # Reward for good coordination
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        """Calculate reward with components for midfield mastery."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "repositioning_reward": [0.0] * len(reward),
            "coordination_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for repositioning: changing position dynamically under opponent pressure
            if np.abs(o['left_team'][o['active']][0] - self.midfield_position) < self.repositioning_reward:
                closest_opponent_distance = np.min(np.sqrt(np.sum((o['left_team'][o['active']] - o['right_team'])**2, axis=1)))
                if closest_opponent_distance < self.pressure_threshold:
                    components["repositioning_reward"][rew_index] = self.repositioning_reward
                    reward[rew_index] += components["repositioning_reward"][rew_index]

            # Reward for coordination: passing under pressure
            if 'sticky_actions' in o:
                if o['sticky_actions'][6] == 1 and closest_opponent_distance < self.pressure_threshold:
                    components["coordination_reward"][rew_index] = self.coordination_reward
                    reward[rew_index] += components["coordination_reward"][rew_index]

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
