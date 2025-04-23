import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for wingers performing crossing and sprinting."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """Modifies the reward based on winger's crossing and sprinting performance."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward),
                      "sprinting_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check for winger in active control
            if o['active'] < 0 or o['left_team_roles'][o['active']] not in (6, 7):  # LM, RM roles
                continue

            # Reward for crossing the ball successfully towards the penalty box
            if o['game_mode'] == 4 and o['ball'][0] > 0.7 and abs(o['ball'][1]) < 0.2:  # Assuming 0.7 as near opponent's goal along x-axis
                components["crossing_reward"][rew_index] = 1.0  # Statically assigned for demonstration

            # Reward for sprinting: Check if sprint action is active
            if o['sticky_actions'][8]:
                components["sprinting_reward"][rew_index] = 0.1  # Increasing reward slightly for sprinting action

            # Calculate total modified reward
            reward[rew_index] += (components["crossing_reward"][rew_index] + components["sprinting_reward"][rew_index])

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
