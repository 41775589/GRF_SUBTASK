import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for mastering midfield dynamics."""

    def __init__(self, env):
        super().__init__(env)
        self.midfield_checkpoints = 0.5  # Position of midfield in normalized coordinates
        self.checkpoint_value = 0.1
        self.checkpoints_collected = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoints_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoints_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # This function assumes the observations are structured with keys that indicate player positions
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_dynamic_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Adjust reward based on player's position in midfield and their strategic actions
        for i, obs in enumerate(observation):
            if 'left_team' in obs:
                # Calculate distance to the midfield and add checkpoint rewards for strategic positioning
                player_pos = obs['left_team'][obs['active']]
                x_dist = abs(player_pos[0] - self.midfield_checkpoints)
                if x_dist < 0.1:  # 10% vicinity threshold for midfield line as strategic areas
                    if i not in self.checkpoints_collected:
                        components["midfield_dynamic_reward"][i] = self.checkpoint_value
                        self.checkpoints_collected[i] = True
                        reward[i] += components["midfield_dynamic_reward"][i]

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
            for j, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[j] = act
        return observation, reward, done, info
