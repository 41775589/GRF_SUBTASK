import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for mastering defensive tactics with tackles."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sliding_tackle_counter = np.zeros(10, dtype=int)
        self.standing_tackle_counter = np.zeros(10, dtype=int)
        self.tackle_efficiency_bonus = 0.15

    def reset(self):
        self.sliding_tackle_counter = np.zeros(10, dtype=int)
        self.standing_tackle_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sliding_tackle_counter'] = self.sliding_tackle_counter
        to_pickle['standing_tackle_counter'] = self.standing_tackle_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sliding_tackle_counter = from_pickle['sliding_tackle_counter']
        self.standing_tackle_counter = from_pickle['standing_tackle_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_tackle_effective = False  # This would come from some logic based on the observations

            # Add logic to determine if a tackle (sliding or standing) was attempted and successful
            if some_condition_for_sliding_tackle(o):  # Define this according to your environment's specifics
                self.sliding_tackle_counter[rew_index] += 1
                if is_tackle_effective:
                    components["tackle_reward"][rew_index] += self.tackle_efficiency_bonus

            if some_condition_for_standing_tackle(o):  # Similarly, define this condition
                self.standing_tackle_counter[rew_index] += 1
                if is_tackle_effective:
                    components["tackle_reward"][rew_index] += self.tackle_efficiency_bonus

            reward[rew_index] += components["tackle_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Reset the sticky_actions_counter logic again if needed, based on specific game conditions.
        return observation, reward, done, info
