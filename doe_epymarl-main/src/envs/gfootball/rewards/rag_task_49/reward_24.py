import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that applies additional rewards for shooting accuracy and power from central field positions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.central_zone_threshold = 0.15  # range around center to define central zone
        self.accuracy_reward = 1.0
        self.power_boost_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "accuracy_reward": [0.0] * len(reward),
            "power_boost_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Centrality check: ball must be near the middle of the field on the x-axis
            if o['ball'][0] > -self.central_zone_threshold and o['ball'][0] < self.central_zone_threshold:
                if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:  # Right team and active player has the ball
                    shot_power = np.linalg.norm(o['ball_direction'][:2])
                    central_factor = (1 - abs(o['ball'][0]/self.central_zone_threshold))

                    # Reward for shooting from central positions with accuracy
                    if shot_power > 0.1:  # assuming a threshold for meaningful shots
                        components["accuracy_reward"][rew_index] = self.accuracy_reward * central_factor

                    # Additional reward for powerful shots
                    if shot_power > 0.5:
                        components["power_boost_reward"][rew_index] = self.power_boost_reward

                    # Combine rewards
                    reward[rew_index] += components["accuracy_reward"][rew_index] + components["power_boost_reward"][rew_index]
                    
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Attach reward info to the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
