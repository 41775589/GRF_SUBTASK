import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on defensive positioning and interceptions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_zones = 5  # Dividing each half into zones
        self._interception_reward = 0.3
        self._positioning_reward = 0.2
        self._collected_zones = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_zones = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_zones
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_zones = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            base_reward = reward[rew_index]
            # Check if the ball is close to being intercepted
            if o['ball_owned_team'] == 1:  # If opposition team has the ball
                if np.linalg.norm(o['ball'] - o['left_team'][o['active']]) < 0.1:
                    components["interception_reward"][rew_index] = self._interception_reward
                    reward[rew_index] += components["interception_reward"][rew_index]
            
            # Reward for being well positioned in their own half defensively
            player_x_pos = o['left_team'][o['active']][0]  
            if player_x_pos <= 0:  # Player is in own half
                zone = int((player_x_pos + 1) / 0.2)  # Normalize position and map to zone
                if zone not in self._collected_zones.get(rew_index, []):
                    components["positioning_reward"][rew_index] = self._positioning_reward
                    reward[rew_index] += components["positioning_reward"][rew_index]
                    if rew_index in self._collected_zones:
                        self._collected_zones[rew_index].append(zone)
                    else:
                        self._collected_zones[rew_index] = [zone]

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
