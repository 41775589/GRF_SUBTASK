import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes mastering midfield dynamics with enhanced coordination and 
    strategic repositioning during transitions between offense and defense."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_checkpoints = np.linspace(-0.5, 0.5, 10)
        self.checkpoint_rewards_collected = {}
        self.midfield_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoint_rewards_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoint_rewards_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoint_rewards_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball' not in o:
                continue

            # Check if the ball is in midfield and manage tactical positioning
            ball_position_x = o['ball'][0]
            collector_count = self.checkpoint_rewards_collected.get(rew_index, 0)

            if abs(ball_position_x) <= 0.5:  # if the ball is in the midfield
                checkpoint_index = np.searchsorted(self.midfield_checkpoints, ball_position_x)
                if collector_count < checkpoint_index:
                    reward[rew_index] += self.midfield_reward
                    components["midfield_reward"][rew_index] += self.midfield_reward
                    self.checkpoint_rewards_collected[rew_index] = checkpoint_index

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
