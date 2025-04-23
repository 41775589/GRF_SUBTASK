import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper for enhancing rewards based on strategic long-range passes and coordination in play.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_threshold = 0.3  # Threshold for considering a pass long-range
        self.passing_reward = 0.05    # Reward increment for successful long passes

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper'].get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        # Base reward
        components = {"base_score_reward": reward.copy(), "long_pass_reward": [0.0, 0.0]}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Team 0 is the team we are training
                if 'action' in o and o['action'] == 9:  # Assuming 9 is the encoding for high or long pass actions
                    # Calculate pass length
                    ball_pos_before = o['ball']
                    ball_pos_after = o['ball'] + o['ball_direction']
                    pass_length = np.linalg.norm(ball_pos_before[0:2] - ball_pos_after[0:2])

                    # If the pass length is above threshold and correctly performed
                    if pass_length > self.passing_threshold:
                        components["long_pass_reward"][idx] += self.passing_reward
                        reward[idx] += 1.5 * components["long_pass_reward"][idx]

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
