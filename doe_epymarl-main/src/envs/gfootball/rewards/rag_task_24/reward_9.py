import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that augments the reward based on the effectiveness of mid to long-range passing,
    emphasizing high and long accurate passes to teammates.
    """

    def __init__(self, env):
        super().__init__(env)
        # Tracking the number of successful long passes
        self.successful_long_passes = [0, 0]
        # We define a long pass at a certain threshold
        self.pass_distance_threshold = 0.2  # Represents a sufficient mid to long range pass
        self.pass_reward_multiplier = 1.5

    def reset(self):
        self.successful_long_passes = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['successful_long_passes'] = self.successful_long_passes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.successful_long_passes = from_pickle.get('successful_long_passes', [0, 0])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]

            # Check for the activation of the long pass action.
            if 'sticky_actions' in o:
                if o['sticky_actions'][5] == 1:  # Assuming index 5 corresponds to long pass action.
                    # Calculating the distance that the ball was passed
                    if ('ball_direction' in o and o['ball_owned_team'] in [0, 1]):
                        direction_norm = np.linalg.norm(o['ball_direction'][:2])

                        if direction_norm > self.pass_distance_threshold:
                            # Reward the agent for making a long pass
                            components["pass_reward"][idx] = self.pass_reward_multiplier * direction_norm
                            reward[idx] += components["pass_reward"][idx]
                            self.successful_long_passes[idx] += 1

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
