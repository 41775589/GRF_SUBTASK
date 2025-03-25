import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for successful defensive maneuvers, specifically for performing
    timely and precise sliding tackles under high-pressure situations.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sliding_tackle_counter = np.zeros(10, dtype=int)  # Tracking the number of sliding tackles
        self.base_reward_for_tackle = 0.5  # Base reward for performing a sliding tackle

    def reset(self):
        # Reset the counter for sliding tackles when the environment is reset
        self.sliding_tackle_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save the state of the tackles counter alongside the state of the environment
        to_pickle['sliding_tackle_counter'] = self.sliding_tackle_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Retrieve and set the saved state of sliding tackle counter
        from_pickle = self.env.set_state(state)
        self.sliding_tackle_counter = from_pickle['sliding_tackle_counter']
        return from_pickle

    def reward(self, reward):
        # Process rewards by considering valued defensive actions
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Add reward if a sliding tackle is successfully performed
            if 'actions' in o:
                if 'action_sliding' in o['actions'] and o['actions']['action_sliding'] > 0:
                    if self.sliding_tackle_counter[rew_index] < 5:  # Limiting the number of rewarded tackles
                        reward[rew_index] += self.base_reward_for_tackle
                        self.sliding_tackle_counter[rew_index] += 1
                        if 'sliding_tackle' not in components:
                            components['sliding_tackle'] = [0.0] * len(reward)
                        components['sliding_tackle'][rew_index] = self.base_reward_for_tackle

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Pass final reward and reward components to info to output to user or further processing
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
