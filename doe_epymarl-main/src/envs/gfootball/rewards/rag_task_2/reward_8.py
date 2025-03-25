import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive strategy coordination reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define distance checkpoints for positioning towards ball ownership areas
        self.positioning_checkpoints = 5
        self.positioning_reward = 0.05  # Weight for controlling segmentation of the field
        self.collected_positioning_rewards = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_positioning_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.collected_positioning_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_positioning_rewards = from_pickle.get('CheckpointRewardWrapper', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "positioning_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_position = np.array(o['right_team'][o['active']])

            # Closer to ball owned by the opposing team
            if o['ball_owned_team'] == 1:
                ball_position = np.array(o['ball'])
                distance_to_ball = np.linalg.norm(current_position - ball_position[:2])

                # Calculate reward based on defensive positioning and ball approach
                checkpoint_index = int((self.positioning_checkpoints * distance_to_ball) // 0.2)
                if checkpoint_index < self.positioning_checkpoints:
                    if self.collected_positioning_rewards.get(rew_index, -1) < checkpoint_index:
                        components["positioning_reward"][rew_index] = self.positioning_reward
                        reward[rew_index] += components["positioning_reward"][rew_index]
                        self.collected_positioning_rewards[rew_index] = checkpoint_index

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
