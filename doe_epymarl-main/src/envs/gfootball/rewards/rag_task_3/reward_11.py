import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward for shooting accurately under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.shot_coefficient = 2.0  # Coefficient for increasing the reward for shots.
        self.pressure_resistance_bonus = 0.5  # Additional reward for resisting pressure.
        self._previous_ball_owned_player = None
        self._previous_ball_owned_team = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_owned_player = None
        self._previous_ball_owned_team = None
        return self.env.reset()
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state
    
    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_reward": [0.0],
                      "pressure_reward": [0.0]}

        if observation is None:
            return reward, components
        
        ball_owned_team = observation[0]['ball_owned_team']
        ball_owned_player = observation[0]['ball_owned_player']

        # Reward shooting efforts, more if under pressure
        if observation[0]['game_mode'] in {6}:  # Assuming game_mode 6 is a shot
            components["shot_reward"][0] = self.shot_coefficient
            reward[0] += self.shot_coefficient
            
            # Check if under pressure, apply extra reward
            if self._previous_ball_owned_team == ball_owned_team and self._previous_ball_owned_player == ball_owned_player:
                components["pressure_reward"][0] = self.pressure_resistance_bonus
                reward[0] += self.pressure_resistance_bonus

        # Update the tracked values
        self._previous_ball_owned_player = ball_owned_player
        self._previous_ball_owned_team = ball_owned_team

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
