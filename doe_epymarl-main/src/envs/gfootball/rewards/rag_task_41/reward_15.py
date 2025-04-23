import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the attacking skills by promoting finishing and creative offensive play."""
    
    def __init__(self, env):
        super().__init__(env)
        self._num_zones = 5 # Number of distinct zones towards the goal
        self._zone_reward = 0.2 # Reward for entering a new zone with possession
        self._zone_thresholds = np.linspace(0.0, 1.0, self._num_zones + 1)[1:-1] # thresholds for each zone
        self._player_in_zone = [0] * self._num_zones # tracking which zone each player has reached
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self._player_in_zone = [0] * self._num_zones
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._player_in_zone
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._player_in_zone = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "zone_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 1 and obs['ball_owned_player'] == obs['active']:  # Check if right team has the ball
                player_x = obs['right_team'][obs['active']][0]  # X position of the ball-carrier
                
                # Determine the current zone of the player
                current_zone = 0
                for idx, threshold in enumerate(self._zone_thresholds):
                    if player_x > threshold:
                        current_zone = idx + 1
                    else:
                        break

                # Update rewards if the player moves to a new higher zone
                if current_zone > self._player_in_zone[i]:
                    components['zone_reward'][i] = self._zone_reward * (current_zone - self._player_in_zone[i])
                    reward[i] += components['zone_reward'][i]
                    self._player_in_zone[i] = current_zone

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
