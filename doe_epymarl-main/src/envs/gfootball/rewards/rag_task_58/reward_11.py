import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward function for mastering defensive coordination and transitioning to attack."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize counters and settings for defensive rewards
        self.defensive_positions_achieved = {}
        self.num_defensive_positions = 5  # Number of positions that contribute to defensive coordination
        self.defensive_reward = 0.2  # Reward for achieving a defensive position
        self.transition_reward = 0.3  # Reward for transitioning from defense to attack
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # For tracking sticky actions usage

    def reset(self):
        """Reset internal state upon starting a new episode."""
        self.sticky_actions_counter.fill(0)
        self.defensive_positions_achieved = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieve state with additional wrapper-defined state for serialization."""
        to_pickle['CheckpointRewardWrapper'] = {
            "defensive_positions_achieved": self.defensive_positions_achieved
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set state from deserialized state, including internal state."""
        from_pickle = self.env.set_state(state)
        self.defensive_positions_achieved = from_pickle['CheckpointRewardWrapper']['defensive_positions_achieved']
        return from_pickle

    def reward(self, reward):
        """Augment the reward based on defensive coordination and secure ball distribution."""
        # Get current observations from the environment
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_coordination_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check for defensive coordination
            if o['active'] in self.defensive_positions_achieved:
                # Give reward once per unique defensive position achieved per episode
                continue

            # Supposing a function is_defensive_pose(o) that checks if a player's position and status match a defensive posture
            if self.is_defensive_pose(o):
                components["defensive_coordination_reward"][rew_index] = self.defensive_reward
                self.defensive_positions_achieved[o['active']] = True
                reward[rew_index] += components["defensive_coordination_reward"][rew_index]

            # Transition rewards - if player moves to offensive after being in a defensive position
            if self.was_in_defense(o) and self.is_now_attacking(o):
                components["transition_reward"][rew_index] = self.transition_reward
                reward[rew_index] += components["transition_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # Apply reward function modifications
        reward, components = self.reward(reward)
        # Update info with new component values
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Extract sticky actions and update the sticky actions counter
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info

    def is_defensive_pose(self, observation):
        """Custom logic to determine if a player is in a defensive pose."""
        # Placeholder logic; will depend on positions and other factors
        return True

    def was_in_defense(self, observation):
        """Check if the player was previously in a defensive position."""
        return True

    def is_now_attacking(self, observation):
        """Check if the player has transitioned to an attacking role."""
        return True
