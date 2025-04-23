import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering sliding tackles during defensive play in FootballEnv."""

    def __init__(self, env):
        super().__init__(env)
        self.game_mode_counter = {}
        self.previous_own_team_prepressure = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions for debugging purposes

    def reset(self):
        self.game_mode_counter = {}
        self.previous_own_team_prepressure = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['game_mode_counter'] = self.game_mode_counter
        state['previous_own_team_prepressure'] = self.previous_own_team_prepressure
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.game_mode_counter = from_pickle['game_mode_counter']
        self.previous_own_team_prepressure = from_pickle['previous_own_team_prepressure']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pressure_defense_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        if 'game_mode' not in observation:
            return reward, components

        # Detecting high-pressure defense situations
        game_mode = observation['game_mode']
        steps_left = observation['steps_left']
        
        # Simulate behavior under high-pressure near defensive third
        high_pressure = game_mode in [2, 3, 4, 6]  # Consider game modes like corner, free-kick against, etc.
        for i in range(len(reward)):
            if high_pressure:
                if observation['ball_owned_team'] == 1 and observation['ball'][0] < 0.3:  # Ball in left defensive third
                    components['pressure_defense_reward'][i] = 0.5
                    reward[i] += components['pressure_defense_reward'][i]
                    self.game_mode_counter[steps_left] = self.game_mode_counter.get(steps_left, 0) + 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
