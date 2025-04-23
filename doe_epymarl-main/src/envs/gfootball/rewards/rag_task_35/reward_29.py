import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic positioning and movement reward to each step."""

    def __init__(self, env):
        super().__init__(env)
        self.max_positioning_repeats = 5
        self.positioning_rewards = {}
        self.positioning_count = {}
        self.positioning_value = 0.05
        self.max_moves_per_game = 500
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the wrapper's internal state when the environment is reset."""
        self.positioning_rewards = {}
        self.positioning_count = {}
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serialize the wrapper's state."""
        to_pickle['positioning_rewards'] = self.positioning_rewards
        to_pickle['positioning_count'] = self.positioning_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Deserialize the wrapper's state."""
        from_pickle = self.env.set_state(state)
        self.positioning_rewards = from_pickle.get('positioning_rewards', {})
        self.positioning_count = from_pickle.get('positioning_count', {})
        return from_pickle

    def reward(self, reward):
        """Add rewards for strategic positioning and movement."""
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy()}
        
        # Initialize reward components for each agent
        for i in range(len(reward)):
            components[f"positional_reward_agent_{i}"] = 0

        # Loop through each agent observation
        for i, o in enumerate(observation):
            # Example strategic positioning based on game situation
            active_position = o['left_team'][o['active']]
            proximity_to_goal = abs(active_position[0] - 0.0)
            
            # Reward positioning close to tactical interests (like ball or goal)
            positioning_reward = np.tanh(1/(proximity_to_goal + 0.1))
            key = (i, tuple(active_position))
            if key not in self.positioning_rewards:
                if self.positioning_count.get(i, 0) < self.max_moves_per_game:
                    reward[i] += self.positioning_value * positioning_reward
                    self.positioning_rewards[key] = self.positioning_value * positioning_reward
                    components[f"positional_reward_agent_{i}"] += self.positioning_value * positioning_reward
                    self.positioning_count[i] = self.positioning_count.get(i, 0) + 1

        return reward, components

    def step(self, action):
        """Process environment step and add custom rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value) if isinstance(value, list) else value
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
