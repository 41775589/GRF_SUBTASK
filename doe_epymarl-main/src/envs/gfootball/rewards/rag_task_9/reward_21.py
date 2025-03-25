import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive football skills such as passing, shooting, and dribbling."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Dictionary to track actions taken by agent leading to positive outcomes
        self.action_rewards = {
            'short_pass': 0.2,
            'long_pass': 0.3,
            'shot': 1.0,
            'dribble': 0.1,
            'sprint': 0.05
        }
        self.action_count = {'short_pass': 0, 'long_pass': 0, 'shot': 0, 'dribble': 0, 'sprint': 0}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.action_count = {'short_pass': 0, 'long_pass': 0, 'shot': 0, 'dribble': 0, 'sprint': 0}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.action_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.action_count = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "action_rewards": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Reward based on action effectiveness: a positively influenced reward mechanism for offensive play
        for rew_index, o in enumerate(observation):
            for key, action_id in zip(['short_pass', 'long_pass', 'shot', 'dribble', 'sprint'], [9, 8, 11, 12, 10]):
                if o['sticky_actions'][action_id]:  # check if action is made
                    # Reward only if the team still owns the ball after the action
                    if o['ball_owned_team'] == 1:  # Assuming the agent's team is always '1'
                        components["action_rewards"][rew_index] += self.action_rewards[key]
                        self.action_count[key] += 1

        # Update rewards list based on enhanced actions
        reward = [(r + com) for r, com in zip(reward, components["action_rewards"])]

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
