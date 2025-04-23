import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards effective mid to long-range passing between players."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define range thresholds for mid and long-range passes.
        self.mid_range_threshold = 0.3  # Mid-range in terms of normalized coordinates
        self.long_range_threshold = 0.6  # Long-range in terms of normalized coordinates
        self.passing_reward = 0.05  # Reward for successful pass
        self.precision_multiplier = 2.0  # Multiplier for precise long-range passes

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            previous_pos = o['ball']
            current_pos = o['ball'] + o['ball_direction']

            # Calculate the Euclidean distance the ball travelled
            dist = np.linalg.norm(current_pos[:2] - previous_pos[:2])

            if dist >= self.mid_range_threshold:
                if o['game_mode'] in {1, 6}:  # 1=KickOff, 6=Penalty, simplified assumptions for controlled game modes
                    # Check if a pass was completed successfully in mid or long range
                    if dist >= self.long_range_threshold:
                        # Extra reward for precision in long-range passes
                        precision = np.clip(np.abs(current_pos[1]), 0, 1)
                        components["passing_reward"][rew_index] += self.precision_multiplier * precision * self.passing_reward
                    else:
                        components["passing_reward"][rew_index] += self.passing_reward
                    
                    reward[rew_index] += components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Calculate updated rewards using the specialized reward function
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add components to the info dict for tracking
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Track sticky actions for information
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}_count"] = self.sticky_actions_counter[i]
                
        return observation, reward, done, info
