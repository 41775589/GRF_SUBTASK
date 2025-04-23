import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for ball control under pressure and strategic play."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize sticky actions counter.

    def reset(self):
        """Reset the sticky actions counter and other necessary elements."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the wrapper."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on control under pressure and strategic play."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "control_under_pressure": [0.0] * len(reward),
                      "strategic_play": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            if obs['ball_owned_team'] == 0:  # Team 0 means the agent's team owns the ball
                # Encourage maintaining ball possession under opponent presence
                proximity_threshold = 0.1
                opponent_distances = np.linalg.norm(obs['right_team'] - obs['ball'][:2], axis=1)
                pressure = np.any(opponent_distances < proximity_threshold)
                if pressure:
                    components['control_under_pressure'][rew_index] += 0.2

                # Reward moving the ball towards less covered areas of the field
                space_utilization = np.mean(np.min(np.linalg.norm(obs['left_team'] - obs['ball'][:2], axis=1)))
                if space_utilization > 0.3:
                    components['strategic_play'][rew_index] += 0.3

            # Update the base reward with additional components
            reward[rew_index] += components['control_under_pressure'][rew_index] + components['strategic_play'][rew_index]

        return reward, components

    def step(self, action):
        """Apply the action and modify the output with the new reward mechanism."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Summarizing reward and components into the info dictionary
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
