import gym
import numpy as np
class LongRangeShootingRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for successful long-range shots, 
    particularly those that beat defenders outside the penalty box."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._long_shot_threshold = 0.6  # Rough estimate threshold beyond the midfield
        self._reward_for_long_shot = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_shot_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Determine if a goal is scored
            if reward[rew_index] == 1:
                # Check if the shot was taken from long range
                if ('ball_owned_team' in o and o['ball_owned_team'] == 0 and
                    'ball' in o and abs(o['ball'][0]) > self._long_shot_threshold):
                    components["long_shot_reward"][rew_index] = self._reward_for_long_shot
                    reward[rew_index] += components["long_shot_reward"][rew_index]

        return reward, components

    def step(self, action):
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
