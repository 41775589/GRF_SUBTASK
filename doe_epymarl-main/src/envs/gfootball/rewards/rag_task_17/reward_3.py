import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards mastering wide midfield responsibilities, such as high passing and positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_bonus = 0.2
        self.position_bonus = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = None
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for successful high pass in a wide position (consider lateral extents)
            if o['sticky_actions'][8] == 1:  # Assuming index 8 corresponds to high pass action
                if abs(o['left_team'][o['active']][1]) > 0.2 or abs(o['right_team'][o['active']][1]) > 0.2:
                    components['pass_reward'][rew_index] = self.pass_bonus
                    reward[rew_index] += components['pass_reward'][rew_index]
            
            # Reward for positioning wide on the field to stretch the defense
            if abs(o['right_team'][o['active']][1]) > 0.3 or abs(o['left_team'][o['active']][1]) > 0.3:
                components['position_reward'][rew_index] = self.position_bonus
                reward[rew_index] += components['position_reward'][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
