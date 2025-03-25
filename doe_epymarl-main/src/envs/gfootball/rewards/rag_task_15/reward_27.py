import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that focuses on enhancing the training specific to mastering long passes:
    including accuracy and understanding the dynamics of ball travel over different lengths.
    """
    def __init__(self, env):
        super().__init__(env)
        self.distance_thresholds = np.linspace(0.2, 0.9, 5)  # Distance thresholds for long passes
        self.pass_rewards = np.linspace(0.1, 0.5, 5)  # Incremental rewards for each threshold
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        """ Reset the sticky actions counter at the beginning of an episode. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Get the state of the environment including states added by the wrapper. """
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set the state of the environment from the state provided. """
        from_pickle = self.env.set_state(state)
        from_pickle['CheckpointRewardWrapper'] = {}
        return from_pickle

    def reward(self, reward):
        """ Reward function that focuses on long passes. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # No extra reward for goals directly
            if reward[rew_index] == 1 or (o['ball'] is None):
                continue

            # Check if the active player has made a long pass
            if o['ball_owned_team'] == 0:
                ball_position_before = np.array(o['ball'][0:2])
                ball_position_after = ball_position_before + np.array(o['ball_direction'][0:2])

                distance_moved = np.linalg.norm(ball_position_after - ball_position_before)

                for idx, threshold in enumerate(self.distance_thresholds):
                    if distance_moved >= threshold:
                        components["long_pass_reward"][rew_index] = self.pass_rewards[idx]
                        reward[rew_index] += components["long_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Wraps the environment's step function to keep track of 
        rewards and add additional information to the 'info' output.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        return observation, reward, done, info
