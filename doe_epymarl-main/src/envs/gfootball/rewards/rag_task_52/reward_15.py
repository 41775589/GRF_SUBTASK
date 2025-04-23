import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards defensive skills, including tackling, efficient positioning to intercept the ball
    and maintaining possession under pressure.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.2
        self.pressure_pass_reward = 0.15
        self.tackle_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CustomRewardWrapper'] = dict()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_reward": [0.0] * len(reward),
                      "pressure_pass_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            original_reward = reward[rew_index]

            # Reward interception if the player is close to ball when the opposing team kicks
            if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball'] - o['left_team'][o['active']]) < 0.1:
                components['interception_reward'][rew_index] = self.interception_reward
                reward[rew_index] += self.interception_reward

            # Reward maintaining possession under pressure
            close_opponents = np.linalg.norm(o['left_team'][o['active']] - o['right_team'], axis=1) < 0.15
            if o['ball_owned_team'] == 0 and any(close_opponents):
                components['pressure_pass_reward'][rew_index] = self.pressure_pass_reward
                reward[rew_index] += self.pressure_pass_reward

            # Reward tackling
            if o['game_mode'] == 2 and o['ball_owned_team'] == 1:  # Assume game_mode 2 is a defensive game state
                components['tackle_reward'][rew_index] = self.tackle_reward
                reward[rew_index] += self.tackle_reward

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
