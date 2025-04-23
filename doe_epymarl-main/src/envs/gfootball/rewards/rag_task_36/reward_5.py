import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on dribbling and dynamic positioning for transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_reward = 0.05
        self.position_change_reward = 0.02

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
        """Adjust the reward based on dribbling and positional changes."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "dribble_reward": [0.0] * 2, "position_change_reward": [0.0] * 2}

        for i, (o, r) in enumerate(zip(observation, reward)):
            dribble_action = o['sticky_actions'][9]
            if dribble_action == 1 and self.sticky_actions_counter[9] == 0:
                components["dribble_reward"][i] = self.dribble_reward
                reward[i] += components["dribble_reward"][i]
            self.sticky_actions_counter[9] = dribble_action

            old_position = self.sticky_actions_counter[10:12]
            current_position = o['left_team'][o['active']][:2] if o['team'] == 'left' else o['right_team'][o['active']][:2]
            
            if np.any(old_position != current_position):
                components["position_change_reward"][i] = self.position_change_reward
                reward[i] += components["position_change_reward"][i]
            
            self.sticky_actions_counter[10:12] = current_position

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
