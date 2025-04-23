import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards actions and positioning that focus on wide midfield responsibilities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define specific regions for wide midfield play
        self.midfield_regions = [(-1, -0.42), (-1, 0), (-0.5, -0.42), (-0.5, 0), (0, -0.42), (0, 0), (0.5, -0.42), (0.5, 0), (1, -0.42), (1, 0)]
        # Rewards for positioning and making high passes in these regions
        self.positional_reward_weight = 0.05
        self.high_pass_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No specific state needed for the wrapper
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positional_reward": [0.0] * len(reward),
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            cur_player_pos = o['left_team'][o['active']] if o['left_team_active'][o['active']] else o['right_team'][o['active']]
            
            # Reward for being in key wide midfield positions
            for region in self.midfield_regions:
                if region[0] - 0.25 <= cur_player_pos[0] <= region[0] + 0.25 and region[1] - 0.25 <= cur_player_pos[1] <= region[1] + 0.25:
                    components["positional_reward"][rew_index] += self.positional_reward_weight
                    reward[rew_index] += self.positional_reward_weight

            # Reward for performing a high pass effectively
            if o['sticky_actions'][6]:  # Assuming index 6 correlates with the high pass action
                components["high_pass_reward"][rew_index] += self.high_pass_reward
                reward[rew_index] += self.high_pass_reward

        return reward, components

    def step(self, action):
        # Execute environment step
        observation, reward, done, info = self.env.step(action)
        
        # Modify the rewards and components based on defined metrics
        modified_reward, reward_components = self.reward(reward)
        
        # Add the modified reward and components details to info for output
        info["final_reward"] = sum(modified_reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
        
        # Print out the sticky actions for debugging purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_value in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action_value
        
        return observation, modified_reward, done, info
