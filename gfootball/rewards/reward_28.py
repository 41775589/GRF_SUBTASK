import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive strategy implementations."""

    def __init__(self, env):
        super().__init__(env)
        self._checkpoint_rewards = {
            'dribbling': 0.2,
            'shooting': 0.5,
            'passing': 0.3
        }

    def reset(self):
        """Resets the environment and any stateful reward tracking variables."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state of the reward wrapper, including collected rewards."""
        to_pickle['CheckpointRewardWrapper'] = self._checkpoint_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state of the reward wrapper from a pickle state."""
        from_pickle = self.env.set_state(state)
        self._checkpoint_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Reward calculation for offensive strategies."""
        observation = self.env.unwrapped.observation()
        components = {
            'base_score_reward': reward.copy(),
            'dribbling_reward': 0.0,
            'shooting_reward': 0.0,
            'passing_reward': 0.0
        }

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Improve dribbling strategy if player has ball and moves
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                distance_moved = np.linalg.norm(o['last_team_direction'])
                if distance_moved > 0.1:  # Assumes player is dribbling
                    reward[i] += self._checkpoint_rewards['dribbling']
                    components['dribbling_reward'] += self._checkpoint_rewards['dribbling']

            # Reward for shooting towards goal
            if o['game_mode'] == 6:  # Assume it's a shooting opportunity
                reward[i] += self._checkpoint_rewards['shooting']
                components['shooting_reward'] += self._checkpoint_rewards['shooting']

            # Reward for successful passes
            if 'successful_pass' in o and o['successful_pass']:
                reward[i] += self._checkpoint_rewards['passing']
                components['passing_reward'] += self._checkpoint_rewards['passing']

        return reward, components

    def step(self, action):
        """Step function with reward shaping."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Write each component into the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
