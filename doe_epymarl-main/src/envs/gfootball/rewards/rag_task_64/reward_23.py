import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on high passes and crossing strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.passed_checkpoints = [False]*5  # Assume some number of checkpoints
        self.checkpoint_rewards = np.linspace(0.1, 0.5, 5)  # Increasing reward values
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.passed_checkpoints = [False]*5
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.passed_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.passed_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        for i in range(len(reward)):
            o = observation[i]

            # Calculate the altitude of the ball which relates to high passes
            ball_z = o['ball'][2]
            checkpoint_idx = min(int(ball_z * 5), 4)  # Scale and discretize ball_z

            if not self.passed_checkpoints[checkpoint_idx]:
                reward[i] += self.checkpoint_rewards[checkpoint_idx]
                self.passed_checkpoints[checkpoint_idx] = True
                components[f"checkpoint_{checkpoint_idx}_reward"] = self.checkpoint_rewards[checkpoint_idx]
            else:
                components[f"checkpoint_{checkpoint_idx}_reward"] = 0.0

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
