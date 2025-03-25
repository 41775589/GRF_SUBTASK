import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic midfield control and defensive maneuvers reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize parameters to manage midfield control and defensive effectiveness
        self._defensive_efficiency = 0.05
        self._midfield_control_value = 0.1

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
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Midfield control rewarding based on ball position and player positions
            if -0.2 <= o['ball'][0] <= 0.2:  # Ball in midfield
                components["midfield_control"] = self._midfield_control_value
                reward[i] += components["midfield_control"]

            # Defensive efficiency based on opponent movements
            adv_nearby = np.any([
                np.linalg.norm(np.array(opponent) - np.array(player)) < 0.1
                for opponent in o['right_team']
                for player in o['left_team']])
            
            if o['game_mode'] in {2, 3, 4} and adv_nearby:  # Goal kick, free kick, or corner
                components["defensive_efficiency"] = self._defensive_efficiency
                reward[i] += components["defensive_efficiency"]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
