import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a transition-focused reward for mastering Sprint, Stop-Sprint, and Dribble actions, especially in counter-attacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize count for sticky actions, specifically targeting sprint and dribble actions

    def reset(self):
        """Reset the sticky action counters and the environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the state of the sticky action counters for game state persistence."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of sticky action counters from the saved state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        """Modify the reward function to additionally focus on sprinting and dribbling actions during rapid transitions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "transition_reward": [0.0]}

        if observation is None:
            return reward, components
        
        # Assuming a single agent
        o = observation[0]
        
        sprint_index = 8  # Index for sprint action
        dribble_index = 9  # Index for dribble action
        
        # Fetch sprint and dribble state
        sprint_active = o['sticky_actions'][sprint_index]
        dribble_active = o['sticky_actions'][dribble_index]

        # Encourage rapid transitions and effective control; higher weights in critical moments
        if sprint_active:
            components['transition_reward'][0] += 0.15  # Encourage sprinting
        if dribble_active and not sprint_active:  # Encourage dribble without sprint for control
            components['transition_reward'][0] += 0.2

        reward[0] += components['transition_reward'][0] * self._reward_acceleration(o)
        
        return reward, components

    def _reward_acceleration(self, o):
        """Calculate dynamic accelerations for rewards based on position and ball control."""
        # E.g., accelerate the reward if near the opponent's goal and owning the ball
        if o['ball_owned_team'] == 0 and o['ball'][0] > 0.5:  # 0.5 as an example threshold
            return 2.0  # Intense focus on attacking plays
        return 1.0

    def step(self, action):
        """Steps the environment, returning observation, reward adjustments, and game status."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
