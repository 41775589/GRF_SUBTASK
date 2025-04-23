import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for precision in long passes."""

    def __init__(self, env):
        super().__init__(env)  # Initialize the parent class
        self.long_pass_threshold = 0.5  # Threshold to consider a pass as 'long'
        self.accuracy_bonus = 0.1  # Additional reward for accurate long passes
        self.initial_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions

    def reset(self):
        """Reset environment and clear the initial ball position."""
        self.initial_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the current state of the wrapper to restore later."""
        to_pickle['CheckpointRewardWrapper'] = {
            'initial_ball_position': self.initial_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the wrapper from saved state."""
        from_pickle = self.env.set_state(state)
        self.initial_ball_position = from_pickle['CheckpointRewardWrapper']['initial_ball_position']
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on the accuracy of long passes."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_accuracy_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check for successful game actions like passes
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                current_ball_position = o['ball'][:2]
                
                if self.initial_ball_position is None:
                    # Capture the initial ball position when it is first owned
                    self.initial_ball_position = current_ball_position
                else:
                    # Calculate distance the ball travelled
                    dist = np.linalg.norm(np.array(current_ball_position) - np.array(self.initial_ball_position))
                    if dist > self.long_pass_threshold:
                        # Check if the end position is near the teammate for accuracy
                        team_positions = o['right_team']
                        close_to_teammate = any(
                            np.linalg.norm(current_ball_position - t_pos[:2]) < 0.1 for t_pos in team_positions
                        )
                        if close_to_teammate:
                            # Add bonus for accurate long pass
                            reward[rew_index] += self.accuracy_bonus
                            components["long_pass_accuracy_bonus"][rew_index] = self.accuracy_bonus
                        
                # Refresh initial position after a long pass or if possession changes
                self.initial_ball_position = current_ball_position if o['ball_owned_team'] == 1 else None

        return reward, components

    def step(self, action):
        """Steps through environment and provides modified rewards with details."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
