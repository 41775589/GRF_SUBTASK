import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for offensive maneuvers and dynamic adaptation during varied game phases."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.previous_game_mode = -1
        self.previous_ball_owned_team = -1
        self.thresh_strike = 0.8  # distance threshold for offensive maneuvers
        self.adaptive_reward_coefficient = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and counters."""
        self.previous_game_mode = -1
        self.previous_ball_owned_team = -1
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the additional attributes of our reward wrapper."""
        to_pickle['previous_game_mode'] = self.previous_game_mode
        to_pickle['previous_ball_owned_team'] = self.previous_ball_owned_team
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the additional attributes from the state."""
        from_pickle = self.env.set_state(state)
        self.previous_game_mode = from_pickle.get('previous_game_mode', -1)
        self.previous_ball_owned_team = from_pickle.get('previous_ball_owned_team', -1)
        return from_pickle

    def reward(self, reward):
        """Enhance the base reward by rewarding offensive maneuvers and dynamic adaptation."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'offensive_maneuver_reward': 0.0, 'adaptation_reward': 0.0}

        if observation is None:
            return reward, components 

        ball_position = observation['ball'][0]  # Only consider x coordinate
        ball_owned_team = observation['ball_owned_team']

        # Reward for offensive maneuvers when the ball is close to the opponent's goal
        if ball_position > self.thresh_strike and ball_owned_team == 1:  # Assuming 1 is the team controlled by the agent
            components['offensive_maneuver_reward'] = 0.1

        # Reward for dynamic adaptation to different game modes
        current_game_mode = observation['game_mode']
        if current_game_mode != self.previous_game_mode:
            if self.previous_game_mode != -1:
                components['adaptation_reward'] = self.adaptive_reward_coefficient
            self.previous_game_mode = current_game_mode

        # Combine components to form final reward
        total_reward = (reward + components['offensive_maneuver_reward'] + components['adaptation_reward'])
        
        return [total_reward], components

    def step(self, action):
        """Process environment step and wrap reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward[0]  # since reward is a single element list
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
