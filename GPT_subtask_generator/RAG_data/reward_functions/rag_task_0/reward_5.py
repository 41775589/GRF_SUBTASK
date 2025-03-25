import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward function by incorporating aspects of 
    offensive strategies including accurate shooting, effective dribbling,
    and break-through passes.
    """

    def __init__(self, env):
        super().__init__(env)
        # Tracks the state of special actions during the episode
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize shooting and dribbling bonuses
        self.shooting_bonus = 0.2
        self.dribbling_bonus = 0.1
        self.passing_bonus = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_bonus": [0.0] * len(reward),
            "dribbling_bonus": [0.0] * len(reward),
            "passing_bonus": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Shooting bonus for goal scoring
            if o['score'][0] > o['score'][1]:  # Assuming the first element in score is for the controlled team.
                components["shooting_bonus"][rew_index] = self.shooting_bonus
                reward[rew_index] += components["shooting_bonus"][rew_index]

            # Dribbling bonus for holding possession and moving forward
            if o['ball_owned_team'] == 0 and o['ball'][0] - o['left_team'][0][0] > 0.05:  # Progress in positive x-direction
                components["dribbling_bonus"][rew_index] = self.dribbling_bonus
                reward[rew_index] += components["dribbling_bonus"][rew_index]

            # Passing bonus for changes in possession within team members without loss
            if o['ball_owned_team'] == 0 and self.sticky_actions_counter[1] > 0:  # Assuming action 1 is related to passing
                components["passing_bonus"][rew_index] = self.passing_bonus
                reward[rew_index] += components["passing_bonus"][rew_index]

            # Update sticky actions used
            self.sticky_actions_counter = o['sticky_actions']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
