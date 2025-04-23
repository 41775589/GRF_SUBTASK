import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that emphasizes mastering wide midfield responsibilities with a focus on high passes and wide positioning."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._high_pass_reward = 0.2
        self._position_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "position_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Rewarding the use of the high pass actively (sticky_actions[9] is high pass)
            if o['sticky_actions'][9] == 1:
                components['high_pass_reward'][rew_index] = self._high_pass_reward
                reward[rew_index] += self._high_pass_reward
            
            # Encouraging the player to be positioned wide on the field
            if abs(o['left_team'][o['active']][1]) > 0.4 or abs(o['right_team'][o['active']][1]) > 0.4:
                components['position_reward'][rew_index] = self._position_reward
                reward[rew_index] += self._position_reward

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
            for i, action_value in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_value
        return observation, reward, done, info
