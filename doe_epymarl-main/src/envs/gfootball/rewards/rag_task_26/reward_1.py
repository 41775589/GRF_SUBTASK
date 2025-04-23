import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds complex midfield dynamics management for reward shaping."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_positions = np.linspace(-0.5, 0.5, 10)
        self.midfield_rewards = np.zeros(10, dtype=int)
        self.midfield_reward_value = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.midfield_positions = np.linspace(-0.5, 0.5, 10)
        self.midfield_rewards = np.zeros(10, dtype=int)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            player_pos = observation[i]['left_team'][observation[i]['active']]
            x_position = player_pos[0]

            for j, midfield_pos in enumerate(self.midfield_positions):
                if abs(x_position - midfield_pos) < 0.05 and self.midfield_rewards[j] == 0:
                    if observation[i]['ball_owned_team'] == 0:  # Ball owned by left team
                        reward[i] += self.midfield_reward_value
                        components["midfield_control_reward"][i] += self.midfield_reward_value
                        self.midfield_rewards[j] = 1
                        break
        
        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.midfield_rewards.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_rewards = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

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
