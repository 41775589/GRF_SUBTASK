import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages attacking skills with specialized rewards for creative offensive play in scenarios with match-like defensive setups."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.offensive_checkpoints = [0.2, 0.4, 0.6, 0.8, 1.0]  # Checkpoints splitting the field
        self.checkpoint_rewards = [0.05, 0.1, 0.15, 0.2, 0.25]  # Increasing rewards
        self.checkpoints_collected = np.zeros((len(env.players), len(self.offensive_checkpoints)))

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoints_collected.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoints_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_play_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, (re, o) in enumerate(zip(reward, observation)):
            # Increment reward if a checkpoint is reached with ball control
            if o['ball_owned_team'] == 1:  # Assuming team 1 is controlled by agents
                player_with_ball = o['ball_owned_player']
                x_position = o['right_team'][player_with_ball, 0]

                for i, checkpoint in enumerate(self.offensive_checkpoints):
                    if x_position > checkpoint and self.checkpoints_collected[rew_index, i] == 0:
                        components["offensive_play_reward"][rew_index] += self.checkpoint_rewards[i]
                        self.checkpoints_collected[rew_index, i] = 1
            
            reward[rew_index] += components["offensive_play_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
