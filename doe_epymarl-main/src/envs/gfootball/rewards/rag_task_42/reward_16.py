import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on midfield dynamics, enhancing coordination and strategic transitions."""

    def __init__(self, env):
        """Initialize the variables for tracking midfield dynamics."""
        super().__init__(env)
        self._midfield_zones = 5  # Number of zones to define in the midfield for checkpoints
        self._zone_reward = 0.05  # Reward for each zone transited actively
        self._strategic_movements = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset sticky actions counter and strategic movements on reset."""
        super().reset()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._strategic_movements = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get current state and include checkpoint data."""
        to_pickle['CheckpointRewardWrapper'] = self._strategic_movements
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state with strategic movements tally."""
        from_pickle = self.env.set_state(state)
        self._strategic_movements = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on strategic movements across midfield zones."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "strategic_movement_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            mid_x = (o['left_team'][:, 0] + o['right_team'][:, 0]).mean()  # Get midfield X coordinate dynamically
            curr_zone = int(mid_x * self._midfield_zones)
            player_zone = int(o['left_team'][o['designated'], 0] * self._midfield_zones)  # Get active player's current zone

            if abs(curr_zone - player_zone) < 2:  # Check if player is actively participating in the midfield gameplay
                key = (rew_index, player_zone)
                if key not in self._strategic_movements:
                    self._strategic_movements[key] = True
                    components["strategic_movement_reward"][rew_index] = self._zone_reward
                    reward[rew_index] += components["strategic_movement_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Step function calls underlying env's step and injects modified reward and components."""
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
