import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards focused on offensive strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define rewards for specific actions and outcomes
        self.shooting_reward = 2.0
        self.dribbling_reward = 0.5
        self.passing_reward = 1.0

    def reset(self):
        """Reset the sticky action counters on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store current state specifics related to our custom rewards."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore state specific to our custom rewards."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Enhance reward function with offensive actions scoring."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": list(reward),   # Keep the base game reward
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate rewards for shooting considering the position and direction
            if o['game_mode'] in [2, 6]:  # Game modes for kicks that might lead to a shot
                if np.linalg.norm(o['ball'] - [1, 0]) < 0.1:  # Close to opponent's goal
                    components["shooting_reward"][rew_index] = self.shooting_reward
                    reward[rew_index] += components["shooting_reward"][rew_index]

            # Reward for dribbling: active if player has the ball and is moving towards opponent's goal
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                if o['ball_direction'][0] > 0:  # Moving towards opponent's goal
                    components["dribbling_reward"][rew_index] = self.dribbling_reward
                    reward[rew_index] += components["dribbling_reward"][rew_index]

            # Reward for passing: considered when the ball is transferred to a teammate closer to opponent's goal
            if o['ball_owned_team'] == 0 and (8 in o['sticky_actions'] or 9 in o['sticky_actions']):  # Pass actions active
                distance_before = np.linalg.norm(o['ball'] - [-1, 0])  # Distance from home goal
                distance_after = np.linalg.norm(o['ball'] + o['ball_direction'] * 10 - [-1, 0])  # Estimate next position
                if distance_after < distance_before:  # Ball is moving towards opponent's goal
                    components["passing_reward"][rew_index] = self.passing_reward
                    reward[rew_index] += components["passing_reward"][rew_index]

        return reward, components

    def step(self, action):
        """Process environment step and calculate our rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Include each reward component in the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
