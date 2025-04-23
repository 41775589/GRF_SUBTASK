import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a denser reward focusing on transitions from defense to attack, specifically emphasizing short pass, long pass, and dribble skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._pass_completion_reward = 0.05
        self._successful_dribble_reward = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "pass_completion_reward": [0.0] * len(reward),
                      "successful_dribble_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for successful pass
            if (o['sticky_actions'][7] or o['sticky_actions'][8]) and o['ball_owned_team'] == o['active']:
                components['pass_completion_reward'][rew_index] = self._pass_completion_reward
            
            # Reward for dribble attempts
            if o['sticky_actions'][9] and o['ball_owned_player'] == o['active']:
                components['successful_dribble_reward'][rew_index] = self._successful_dribble_reward
            
            total_additional_rewards = (components['pass_completion_reward'][rew_index] + 
                                        components['successful_dribble_reward'][rew_index])
            
            reward[rew_index] += total_additional_rewards

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
