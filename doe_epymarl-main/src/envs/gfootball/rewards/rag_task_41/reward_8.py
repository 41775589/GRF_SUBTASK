import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on attack enhancement in football gameplay."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._attack_zones = 6  # Divide the opponent's half into 6 zones
        self._zone_reward = 0.05  # Reward for entering a new zone with possession of the ball
        self._scored_goal_reward = 2  # Higher reward for scoring a goal
        self._progressive_play_reward = 0.1  # Reward for moving the ball forward in the opponent's half
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
                      "attack_zone_reward": [0.0] * len(reward),
                      "progressive_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                x_position = o['ball'][0]

                # Determine the zone of the ball
                zone_index = int((x_position + 1) * self._attack_zones / 2)
                zone_key = (rew_index, zone_index)

                # Reward for entering new attack zones with the ball
                if zone_key not in self._collected_zones:
                    self._collected_zones[zone_key] = True
                    components["attack_zone_reward"][rew_index] = self._zone_reward

                # Additional rewards for progressive play
                if x_position > 0:
                    components["progressive_play_reward"][rew_index] = self._progressive_play_reward

                reward[rew_index] += components["attack_zone_reward"][rew_index] + components["progressive_play_reward"][rew_index]

            # Reward for scoring a goal
            if 'score' in o and o['score'][1] > o['score'][0]:  # Assuming team 1 is the right team
                components["base_score_reward"][rew_index] = self._scored_goal_reward

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
