import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for teamwork and coordination in defensive strategies."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define some checkpoints based on defensive positions
        self.collected_checkpoints = {}
        self.num_checkpoints = 5
        self.checkpoint_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Calculate Euclidean distance from own goal to evaluate defensive strategy
            x, y = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            goal_y = 0  # Y coordinate of the goal is 0 (center of the goal)
            goal_x = -1 if o['ball_owned_team'] == 0 else 1  # X coordinate depends on the team
            distance = np.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2)

            while (self.collected_checkpoints.get(rew_index, 0) < self.num_checkpoints):
                reward_threshold = (2 + 0.3 * (self.num_checkpoints - self.collected_checkpoints.get(rew_index, 0)))
                if distance > reward_threshold:
                    break
                components["checkpoint_reward"][rew_index] = self.checkpoint_reward
                reward[rew_index] += 1.5 * components["checkpoint_reward"][rew_index]
                self.collected_checkpoints[rew_index] = self.collected_checkpoints.get(rew_index, 0) + 1

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
