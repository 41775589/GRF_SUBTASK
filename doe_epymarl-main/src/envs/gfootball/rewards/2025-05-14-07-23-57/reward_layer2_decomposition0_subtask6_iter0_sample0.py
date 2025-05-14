import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper designed to specifically focus on improving sliding techniques and reactions to physical confrontations in a football game scenario."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sprint_count = 0
        self.sliding_actions_counter = 0
        self.physical_confrontations_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sprint_count = 0
        self.sliding_actions_counter = 0
        self.physical_confrontations_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sprint_count': self.sprint_count,
            'sliding_actions_counter': self.sliding_actions_counter,
            'physical_confrontations_counter': self.physical_confrontations_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.sprint_count = state_info['sprint_count']
        self.sliding_actions_counter = state_info['sliding_actions_counter']
        self.physical_confrontations_counter = state_info['physical_confrontations_counter']
        return from_pickle

    def reward(self, reward):
        # Base score reward
        components = {"base_score_reward": reward.copy()}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for o in observation:
            # Increase punishment if not using sliding in potential physical confrontations
            if o['game_mode'] in [2, 3, 4, 6]:  # Potential game modes that can involve physical confrontations
                if o['sticky_actions'][6] == 1:  # Sliding is index 6 of sticky actions
                    reward[0] += 2.0  # Reward for using slide in confrontations
                    self.sliding_actions_counter += 1
                else:
                    reward[0] -= 1.0  # Penalty for not using slide when needed
                self.physical_confrontations_counter += 1

        return reward, {"base_score_reward": reward.copy(), "sliding_rewards": self.sliding_actions_counter, "physical_confrontation_rewards": self.physical_confrontations_counter}

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
