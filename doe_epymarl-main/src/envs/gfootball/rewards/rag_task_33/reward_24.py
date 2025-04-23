import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a custom reward for long-distance shooting with considerations of defensive pressure.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_shot_multiplier = 0.8  # Reward multiplier for attempting long shots
        self.defensive_pressure_penalty = -0.2  # Penalty for shooting under high pressure

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
        """
        Modify the reward based on the agents' actions, encouraging them to shoot from distance and manage defensive pressure.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_shot_bonus": 0.0,
                      "pressure_penalty": 0.0}

        if observation is None or 'ball' not in observation or 'left_team' not in observation:
            return reward, components

        ball_x, ball_y = observation['ball'][0], observation['ball'][1]
        left_team = observation['left_team']
        right_team = observation['right_team']

        # Check if in shooting range (outside penalty box and in the opponent's half)
        if ball_x > 0 and abs(ball_y) > 0.2:
            pressure = 0.0
            for player in right_team:
                distance = np.sqrt((player[0]-ball_x) ** 2 + (player[1]-ball_y) ** 2)
                if distance < 0.1:
                    pressure += 1

            if observation['active'] == observation['ball_owned_player']:
                reward += self.long_shot_multiplier
                components['long_shot_bonus'] += self.long_shot_multiplier

            # Apply defensive pressure penalty if there are more than 2 defenders within a close range
            if pressure >= 2:
                reward += self.defensive_pressure_penalty * pressure
                components['pressure_penalty'] += self.defensive_pressure_penalty * pressure

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
