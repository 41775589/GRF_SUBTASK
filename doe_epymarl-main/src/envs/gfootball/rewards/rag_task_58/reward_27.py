import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper custom-designed to enhance agents' defensive coordination skills
       by providing additional rewards for successful defensive actions followed 
       by efficient ball distribution and transitioning to attack."""

    def __init__(self, env):
        super().__init__(env)
        self._num_defensive_zones = 6  # Number of defensive zones to monitor
        self._defensive_rewards = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]  # Reward for each zone
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize metrics to track the ball in specific zones
        self.defensive_position_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_position_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.defensive_position_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_position_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for defending effectively in certain zones close to own goal
            defending_position_reward = np.interp(
                    o['ball'][0], [-1, -0.5], [0.1, 0]) * (o['ball_owned_team'] == 0)

            player_in_def_zone = o['ball'][0] < -0.5 and o['ball_owned_team'] == 0
            zone_to_reward_percentage = int(abs(o['ball'][0] + 1) / 0.2)
            
            if player_in_def_zone:
                current_zone_reward = self._defensive_rewards[min(zone_to_reward_percentage, len(self._defensive_rewards) - 1)]
                components["defensive_reward"][rew_index] = current_zone_reward
                reward[rew_index] += current_zone_reward

            components["defensive_reward"][rew_index] = defending_position_reward
            reward[rew_index] += defending_position_reward

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
