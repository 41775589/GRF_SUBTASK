import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This reward wrapper emphasizes on defensive actions such as tackling, 
    intercepting the ball, and transitioning to an effective attack by successful passes.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercept_reward = 0.1
        self.tackle_reward = 0.1
        self.positive_pass_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "intercept_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward),
                      "positive_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
            
        for rew_index, o in enumerate(observation):
            # Defensive rewards
            if o['game_mode'] == 3:  # FreeKick to opposing team, likely after intercept or tackle
                components['intercept_reward'][rew_index] = self.intercept_reward
                components['tackle_reward'][rew_index] = self.tackle_reward
                reward[rew_index] += self.intercept_reward + self.tackle_reward

            # Transition to attack rewards (measured by possession regaining and forward ball movement by passes)
            if o['ball_owned_team'] == 0 and np.linalg.norm(o['ball_direction'][:2]) > 0:
                components['positive_pass_reward'][rew_index] = self.positive_pass_reward
                reward[rew_index] += self.positive_pass_reward

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
