import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards successful long-range shots and positions favorable for such shots."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.long_shot_bonus = 1.0  # Additional reward for scoring from long distance
        # Initializes tracking for actions leading to long shots
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # For tracking actions like sprint, dribble

    def reset(self):
        """Reset environment and the counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include state of the sticky actions for continuity in serialized environments."""
        to_pickle['sticky_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore state from serialized environment."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions'])
        return from_pickle

    def reward(self, reward):
        """Custom reward function to encourage long-range shots."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'long_shot_bonus': [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 1 and o['ball'][0] > 0.5 and abs(o['ball'][1]) < 0.07:
                # The player is in a good position for a long-range goal
                components['long_shot_bonus'][rew_index] = self.long_shot_bonus
                reward[rew_index] += components['long_shot_bonus'][rew_index]
            reward[rew_index] += components['base_score_reward'][rew_index]

        return reward, components
                    
    def step(self, action):
        """Steps environment, adjusts reward, and adds info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
