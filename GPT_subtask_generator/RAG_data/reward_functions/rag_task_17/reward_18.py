import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a checkpoint reward focused on wide midfield play, high passes, and positioning."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_usage = 0
        
        # Specific checkpoints related to midfield wide play
        self.midfield_checkpoints = [
            [-0.4, 0.4], [0.0, 0.4], [0.4, 0.4],  # Top wide positions
            [-0.4, 0.0], [0.0, 0.0], [0.4, 0.0],  # Middle wide positions
            [-0.4, -0.4], [0.0, -0.4], [0.4, -0.4] # Bottom wide positions
        ]
        self.checkpoint_counter = np.zeros(len(self.midfield_checkpoints), dtype=int)

    def reset(self):
        """Reset reward wrapper state for a new episode."""
        self.sticky_actions_counter.fill(0)
        self.checkpoint_counter.fill(0)
        self.high_pass_usage = 0
        return self.env.reset()

    def reward(self, reward):
        """Compute additional reward for mastering wide midfield gameplay."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = np.array(o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']])
            
            for idx, checkpoint in enumerate(self.midfield_checkpoints):
                if self.checkpoint_counter[idx] == 0 and np.linalg.norm(player_pos - checkpoint) < 0.1:
                    components["positioning_reward"][rew_index] += 0.1
                    self.checkpoint_counter[idx] = 1
            
            # Reward for effective high passes
            if o['sticky_actions'][6] == 1:  # Action high pass is detected as activated
                components["passing_reward"][rew_index] += 0.05
                self.high_pass_usage += 1

        for rw_index in range(len(reward)):
            reward[rw_index] += components["positioning_reward"][rw_index] + components["passing_reward"][rw_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
