import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for shooting from specific central field positions with high accuracy and power."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._position_checkpoint_reached = False
        self._shooting_distance_score = 0.5
        self._accuracy_bonus = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._position_checkpoint_reached = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'position_checkpoint_reached': self._position_checkpoint_reached
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        pickle_data = from_pickle['CheckpointRewardWrapper']
        self._position_checkpoint_reached = pickle_data.get('position_checkpoint_reached', False)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_position_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            # Check if player is in the desired central shooting zone
            if (not self._position_checkpoint_reached and 
                -0.15 <= o['ball'][0] <= 0.15 and 
                -0.042 <= o['ball'][1] <= 0.042):
                self._position_checkpoint_reached = True
                components["shooting_position_reward"][i] += self._shooting_distance_score

            # Additional bonus for high accuracy shots towards the goal
            if self._position_checkpoint_reached and o['game_mode'] == 6:  # Assuming mode 6 is a shot attempt
                components["shooting_position_reward"][i] += self._accuracy_bonus

            reward[i] += components["shooting_position_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
