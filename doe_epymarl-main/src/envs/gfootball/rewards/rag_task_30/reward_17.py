import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds strategic positioning and transition rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward scaling factors based on strategic events
        self.ball_retreat_penalty = -0.05
        self.counter_attack_reward = 0.1
        self.positioning_reward = 0.05
        self.last_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['latitude'] = self.last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle.get('latitude', None)
        return from_pickle

    def reward(self, reward):
        obs = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": 0.0,
                      "counter_attack_reward": 0.0,
                      "ball_retreat_penalty": 0.0}

        if obs is None or self.last_ball_position is None:
            return reward, components

        ball_position = obs['ball'][0]  # Assuming this is the X position

        if self.last_ball_position:
            movement_x = ball_position - self.last_ball_position

            # Encourage moving forward rapidly after gaining possession (Counter Attack)
            if obs['ball_owned_team'] == 1 and movement_x > 0:
                reward += self.counter_attack_reward
                components['counter_attack_reward'] = self.counter_attack_reward

            # Penalize retreating with the ball toward own goal
            elif obs['ball_owned_team'] == 1 and movement_x < 0:
                reward += self.ball_retreat_penalty
                components['ball_retreat_penalty'] = self.ball_retreat_penalty
            
            # Positioning reward when moving sideways in own half to open up play
            if ball_position < 0 and abs(movement_x) <= 0.05:
                reward += self.positioning_reward
                components['positioning_reward'] = self.positioning_reward

        self.last_ball_position = ball_position
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
