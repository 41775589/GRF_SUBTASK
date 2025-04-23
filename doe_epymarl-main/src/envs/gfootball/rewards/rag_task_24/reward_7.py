import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to enhance mid to long-range passing effectiveness in a football game."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_threshold = 0.3  # Threshold for considering a pass as mid to long-range
        self.precision_reward = 0.1  # Reward for precise passing

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        # We focus only on individual agent rewards, hence length should always be 2
        assert len(reward) == len(observation)

        # Initialize components dictionary
        components = {"base_score_reward": reward.copy(),
                      "precision_passing_reward": [0.0, 0.0]}
        
        # Evaluate each agent's observation
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Only add rewards if the ball is owned by the current agent's team
            if o['ball_owned_team'] == o['active'] and o['ball_owned_team'] != -1:
                last_ball_position = o['ball']
                last_player_position = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
                
                # Calculate distance ball was passed based on current and prior observations
                # Storing last observations would be needed in real-case scenario, here we only mock up
                pass_distance = np.linalg.norm(o['ball'] - last_ball_position)

                # Consider it as a potential rewardable pass if above the threshold and precision is high
                if pass_distance > self.passing_threshold:
                    precision_metric = np.linalg.norm(o['ball_direction'])  # Simplified precision estimation
                    if precision_metric < 0.05:  # Threshold for considering a precision high
                        components['precision_passing_reward'][rew_index] = self.precision_reward
                        reward[rew_index] += self.precision_reward

        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
