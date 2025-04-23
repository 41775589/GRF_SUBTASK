import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym reward wrapper that focuses on enhancing advanced ball control and passing skills under pressure.
    The wrapper provides rewards for mastering Short Pass, High Pass, and Long Pass during tight game situations.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward coefficients for different types of passes
        self.pass_reward_coefficients = {
            'short_pass': 0.1,   # Encourage short passes
            'high_pass': 0.2,    # Higher reward for high passes due to complexity
            'long_pass': 0.3     # Highest reward for long passes
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
    
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if the player has possession and is in a tight position
            if (o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']):
                opponent_proximity = np.min(np.linalg.norm(o['left_team'] - o['right_team'][o['active']], axis=1))
                
                # Tight position defined as opponent within 0.1 units
                if opponent_proximity < 0.1:

                    # Check sticky actions for type of pass executed
                    if o['sticky_actions'][3]:  # Assuming 3 is High Pass
                        components["pass_reward"][rew_index] += self.pass_reward_coefficients['high_pass']
                    elif o['sticky_actions'][1]:  # Assuming 1 is Short Pass
                        components["pass_reward"][rew_index] += self.pass_reward_coefficients['short_pass']
                    elif o['sticky_actions'][2]:  # Assuming 2 is Long Pass
                        components["pass_reward"][rew_index] += self.pass_reward_coefficients['long_pass']

            # Update reward based on the components
            reward[rew_index] += components["pass_reward"][rew_index]

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
