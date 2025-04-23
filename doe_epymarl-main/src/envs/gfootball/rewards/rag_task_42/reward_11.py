import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering midfield dynamics including enhanced coordination under pressure and strategic repositioning for offense and defense transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_checkpoints = 10
        self.midfield_checkpoint_reward = 0.05

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
                      "midfield_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            if 'left_team' in o:
                # Compute midfield region control
                player_positions = np.concatenate((o['left_team'], o['right_team']))
                midfield_zone_indices = np.where((player_positions[:, 0] > -0.3) & (player_positions[:, 0] < 0.3))[0]
                midfield_control = len(np.where(midfield_zone_indices < len(o['left_team']))[0]) - len(
                    np.where(midfield_zone_indices >= len(o['left_team']))[0])

                # Reward tactics at midfield transitions
                components['midfield_reward'][i] = midfield_control * self.midfield_checkpoint_reward
                reward[i] += components['midfield_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
