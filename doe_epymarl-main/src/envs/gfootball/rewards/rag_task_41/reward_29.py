import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for creative attacking play and finishing."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._finish_reward = 1.0
        self._creative_play_reward = 0.5
        self._num_creative_zones = 8
        self._collected_zones = {}

    def reset(self):
        self._collected_zones = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
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
                      "finish_reward": [0.0] * len(reward), 
                      "creative_play_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Assign reward when the ball is controlled near the opponent's goal and a goal is scored
            if o['score'][1] > o['score'][0]:
                components["finish_reward"][rew_index] += self._finish_reward
                reward[rew_index] += components["finish_reward"][rew_index]

            ball_possession = o.get('ball_owned_team', -1) == 1
            active_player = o.get('active')
            if ball_possession and o.get('ball_owned_player', -1) == active_player:
                zone_id = np.clip((o['ball'][0] + 1) * 4, 0, self._num_creative_zones-1)
                if zone_id not in self._collected_zones.get(rew_index, {}):
                    self._collected_zones.setdefault(rew_index, {})[zone_id] = True
                    components["creative_play_reward"][rew_index] += self._creative_play_reward
                    reward[rew_index] += components["creative_play_reward"][rew_index]

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
