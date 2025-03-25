import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focusing on advanced dribbling with effective usage of sprint actions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._num_zones = 5  # Divide the field into 5 horizontal zones
        self._sprint_reward = 0.05  # Reward for sprinting in strategic zones
        self._dribble_reward = 0.03  # Reward for effective dribbling
        self._zone_width = 2.0 / self._num_zones
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sprint_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            zone_idx = int((o['ball'][0] + 1) / self._zone_width)
            
            # Check for sprint action in specific zones (offensive zones, index 3 and 4)
            if o['sticky_actions'][8] == 1 and zone_idx in [3, 4]:
                components["sprint_reward"][rew_index] = self._sprint_reward
                reward[rew_index] += components["sprint_reward"][rew_index]

            # Reward for dribbling while avoiding defenders effectively in any zone
            if o['sticky_actions'][9] == 1:
                # Assume nearest opponent calculation is available via some metric
                # Simulating proximity check: the closer the opponent, higher the dribble reward
                opponent_distance = np.min(np.sqrt(np.sum(np.square(o['right_team'] - o['ball'][:2]), axis=1)))
                if opponent_distance < 0.1:  # Proximity threshold for higher skill dribbling
                    components["dribble_reward"][rew_index] = self._dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
