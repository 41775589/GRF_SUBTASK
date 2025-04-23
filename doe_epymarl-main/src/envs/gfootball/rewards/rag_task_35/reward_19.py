import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that encourages maintaining strategic positioning,
    using all directional movements, and managing transitions between defensive and attacking stances."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.reset()

    def reset(self):
        """Reset for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state for serialization."""
        to_pickle['CheckpointRewardWrapper'] = self.env.get_state(to_pickle)
        return to_pickle

    def set_state(self, state):
        """Set the state from deserialization."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Compute and return the enhanced reward based on strategic positioning and transitions."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "positioning_bonus": [0.0, 0.0]
        }

        for rew_index, o in enumerate(observation):
            # Reward for maintaining strategic position depending on game mode
            if o['game_mode'] == 0:
                # Reward the agent for being in a good attacking or defensive position depending on ball possession
                if o['ball_owned_team'] == o['active']:
                    # Attack mode
                    dist_to_goal = np.abs(o['ball'][0] - 1)  # distance to opponent's goal
                    components['positioning_bonus'][rew_index] = max(0, 0.1 * (1 - dist_to_goal))
                else:
                    # Defensive mode
                    dist_to_own_goal = np.abs(o['ball'][0] + 1)  # distance to own goal
                    components['positioning_bonus'][rew_index] = max(0, 0.1 * (1 - dist_to_own_goal))

            # Reward for using varied actions
            unique_actions = set(o['sticky_actions'])
            if len(unique_actions) > 1:
                components['positioning_bonus'][rew_index] += 0.05

            # Apply calculated components to original reward
            reward[rew_index] += components['positioning_bonus'][rew_index]

        return reward, components

    def step(self, action):
        """Override the environment's step function to include reward computation."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
