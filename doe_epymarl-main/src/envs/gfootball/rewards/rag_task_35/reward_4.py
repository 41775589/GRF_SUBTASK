import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.position_checkpoints = 10
        self.position_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.collected_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_positions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward.copy(), "position_reward": [0.0, 0.0]}

        base_reward = reward.copy()      
        position_rewards = [0.0] * len(reward)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            assert 'ball_owned_team' in obs and 'active' in obs
            
            if obs['ball_owned_team'] == 0:  # Ball is with the left team
                position = obs['left_team'][obs['active']]
            elif obs['ball_owned_team'] == 1:  # Ball is with the right team
                position = obs['right_team'][obs['active']]
            else:
                continue  # Ball is not owned, skip

            position_x_quota = int((position[0] + 1) / 2 * self.position_checkpoints)
            checkpoint_key = (rew_index, position_x_quota)

            if checkpoint_key not in self.collected_positions:
                self.collected_positions[checkpoint_key] = True
                position_rewards[rew_index] = self.position_reward
                reward[rew_index] += position_rewards[rew_index]

        return reward, {"base_score_reward": base_reward, "position_reward": position_rewards}

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        return observation, reward, done, info
