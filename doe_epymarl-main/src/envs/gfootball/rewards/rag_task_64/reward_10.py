import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards high passes and crossings from various distances and angles,
    focusing on dynamic attacking plays and spatial creation for the team.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.crossing_checkpoints = [[0.8, 0.2], [0.8, -0.2], [1.0, 0.1], [1.0, -0.1]]
        self.crossing_distances = [0.3, 0.5, 0.7]
        self.pass_height_bonus = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Set a save state for the environment."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the saved state."""
        from_pickle = self.env.set_state(state)
        # No state to restore as it does not maintain inter-step state,
        # but needed for restoration structure.
        return from_pickle

    def reward(self, reward):
        """
        Augment reward based on the intended task of high passing and crossing,
        focusing on specific areas of the field and ball elevation.
        """
        observation = self.env.unwrapped.observation()
        additional_rewards = [0.0] * len(reward)
        
        if observation is None:
            return reward, {'base_score_reward': reward.copy(), 'pass_height_reward': additional_rewards}

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_z = o['ball'][2]  # Extract the height z of the ball

            # Reward for high passes
            if ball_z > 0.15:  # Suppose a threshold for height that considers a pass 'high'
                additional_rewards[rew_index] += self.pass_height_bonus

                # Calculate distance from key crossing positions
                ball_pos = np.array(o['ball'][:2]) # Current 2D position [x, y] of the ball
                for point in self.crossing_checkpoints:
                    distance = np.linalg.norm(ball_pos - np.array(point))
                    for dist_threshold in self.crossing_distances:
                        if distance < dist_threshold:
                            additional_rewards[rew_index] += 0.05  # Incremental reward for being within crossing distance

            reward[rew_index] += additional_rewards[rew_index]

        return reward, {'base_score_reward': reward.copy(), 'pass_height_reward': additional_rewards}

    def step(self, action):
        """Step function to propagate the action into the environment."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        
        # Sum component rewards for information
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Also pass the counts of the sticky actions taken so far,
        # Needed for complete info updates and introspection
        obs_unwrapped = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs_unwrapped:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return obs, reward, done, info
