import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances strategy by rewarding advanced positioning and reactive passing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_advanced_coefficient = 0.2
        self.reactive_passing_coefficient = 0.3
        self.defensive_formation_coefficient = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_advanced_reward": [0.0] * len(reward),
            "reactive_passing_reward": [0.0] * len(reward)
        }

        if observation is None or len(observation) == 0:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            obs = observation[idx]
            ball_position = obs['ball'][0]  # x-coordinate of the ball

            # Enhancing forward ball movement by the team
            if obs['ball_owned_team'] == 0:  # Assuming left team is our team
                if ball_position > 0:  # Ball is in opponent's half
                    components["ball_advanced_reward"][idx] = self.ball_advanced_coefficient * (ball_position)
                    reward[idx] += components["ball_advanced_reward"][idx]

            # Increased rewards for reactive passing in challenging scenarios
            if obs['game_mode'] in {3, 4, 6}:  # FreeKick, Corner, Penalty
                components["reactive_passing_reward"][idx] = self.reactive_passing_coefficient
                reward[idx] += components["reactive_passing_reward"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
