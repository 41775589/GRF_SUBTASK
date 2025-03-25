import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful and timely defensive actions with sliding tackles in high-pressure scenarios."""

    def __init__(self, env):
        super().__init__(env)
        # Counter for each player's successful sliding tackles
        self.sliding_tackles_counter = np.zeros((2, 5), dtype=int)  # Assuming a 5v5 game
        self.tackle_success_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Thresholds for what's considered a "timely" tackle
        self.time_threshold = 100  # steps allowed after high-pressure indication

        # Timing window tracking when high pressure is applied
        self.high_pressure_window = np.full((2, 5), -1, dtype=int)

    def reset(self):
        # Reset counters and states
        self.sliding_tackles_counter = np.zeros((2, 5), dtype=int)
        self.high_pressure_window = np.full((2, 5), -1, dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sliding_tackles_counter'] = self.sliding_tackles_counter
        to_pickle['high_pressure_window'] = self.high_pressure_window
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sliding_tackles_counter = from_pickle['sliding_tackles_counter']
        self.high_pressure_window = from_pickle['high_pressure_window']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "tackle_timing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if 'game_mode' in o and o['game_mode'] != 0:  # Only adjust rewards during normal play
                continue
            
            if o['ball_owned_team'] in (0, 1):  # Team 0 or 1 has the ball
                team_idx = o['ball_owned_team']
                player_idx = o['ball_owned_player']

                if self.sliding_tackles_counter[team_idx, player_idx] < 5:  # Limit sliding tackles tracked
                    defensive_pressure = (o['opponents_in_proximity'] > 3)  # Define high pressure situation
                    current_step = o['steps_left']

                    if defensive_pressure:
                        if self.high_pressure_window[team_idx, player_idx] == -1:
                            self.high_pressure_window[team_idx, player_idx] = current_step
                    
                    if self.high_pressure_window[team_idx, player_idx] > 0 and current_step - self.high_pressure_window[team_idx, player_idx] <= self.time_threshold:
                        # If the player performs a sliding tackle within time limits
                        if 'sliding_tackle' in o['action_types_performed']:
                            components['tackle_timing_reward'][rew_index] = self.tackle_success_reward
                            reward[rew_index] += self.tackle_success_reward
                            self.sliding_tackles_counter[team_idx, player_idx] += 1
                            # Reset the timing window
                            self.high_pressure_window[team_idx, player_idx] = -1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Adding final reward and components to info for diagnostics
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        
        # Reset sticky actions counter and provide sticky action details in info
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
