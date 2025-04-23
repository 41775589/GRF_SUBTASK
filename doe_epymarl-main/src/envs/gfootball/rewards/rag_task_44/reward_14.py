import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that adds a targeted reward for successfully performing
    the Stop-Dribble maneuver under pressure and quickly changing to a stationary
    position as a defensive tactic.
    """
    def __init__(self, env):
        super().__init__(env)
        # Counter for stop-dribble actions that help avoid pressure.
        self.stop_dribble_counter = 0
        self.pressure_threshold = 0.2  # Proximity to consider pressure.
        self._stop_dribble_reward = 0.5  # Reward for a stop-dribble under pressure.
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.stop_dribble_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'stop_dribble_counter': self.stop_dribble_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.stop_dribble_counter = from_pickle['CheckpointRewardWrapper']['stop_dribble_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "stop_dribble_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Check for pressure and stop-dribble action.
            if o['sticky_actions'][9] and o['ball_owned_team'] == o['active']:
                # Assume proximity to opponents signifies pressure.
                ball_position = o['ball']
                opponent_positions = o['right_team'] if o['active'] == 0 else o['left_team']
                under_pressure = any(
                    np.linalg.norm(ball_position - position) < self.pressure_threshold
                    for position in opponent_positions
                )

                if under_pressure:
                    self.stop_dribble_counter += 1
                    components["stop_dribble_reward"][rew_index] = self._stop_dribble_reward
        
        # Update reward considering the stop-dribble performance while under pressure.
        for i in range(len(reward)):
            reward[i] += components["stop_dribble_reward"][i]
        
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
