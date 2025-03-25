import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides a reward based on wide midfield responsibilities focusing on 
    high pass and lateral positioning. The reward emphasizes mastering the use of wide plays
    to stretch the opponent's defense and create spaces by accurately positioning and passing.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0] * len(reward),
            "pass_quality_reward": [0.0] * len(recovery_metric)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for wide positioning
            own_x_coords = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
            max_y_coord = np.max(np.abs(own_x_coords[:, 1]))  # Extract max y-axis value

            # Simple positional reward based on y spread
            components["positioning_reward"][rew_index] = max_y_coord

            # High pass quality reward
            if 'high_pass' in o['sticky_actions'] and o['sticky_actions'][o['active']] == True:
                components["pass_quality_reward"][rew_index] = 0.3  # Reward high pass

            # Combine components to calculate the final rewards
            reward[rew_index] += components["positioning_reward"][rew_index] + components["pass_quality_reward"][rew_index]

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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
