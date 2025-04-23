import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward structure tailored for training a goalkeeper agent.
    It incentives stopping shots, quickly deciding ball distribution under pressure,
    and effective communication with defenders.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shot_stops": [0.0] * len(reward),
            "quick_decision": [0.0] * len(reward),
            "communication_bonus": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for i, rew in enumerate(reward):
            obs = observation[i]

            # Reward for shot stopping
            if obs['game_mode'] == 6:  # Penalty kick
                if obs['ball_owned_team'] == 0 and obs['ball'][0] < -0.8:
                    components['shot_stops'][i] += 1.0

            # Quick decision-making under pressure
            if obs['game_mode'] in [3, 4, 5]:  # Freekick, Corner, or Throw-In
                if obs['ball_owned_team'] == 0:
                    if np.sum(self.sticky_actions_counter) < 3:  # Assumed quick decision making
                        components['quick_decision'][i] += 0.5

            # Communication with defenders
            own_team_positions = obs['left_team'] if obs['ball_owned_team'] == 0 else obs['right_team']
            close_defenders = np.sum(np.linalg.norm(own_team_positions - obs['ball'][:2], axis=1) < 0.1)
            components['communication_bonus'][i] += 0.1 * close_defenders

            # Update rewards with components
            reward[i] += components['shot_stops'][i] + components['quick_decision'][i] + components['communication_bonus'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
