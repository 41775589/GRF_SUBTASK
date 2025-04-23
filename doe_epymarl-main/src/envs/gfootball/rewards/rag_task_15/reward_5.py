import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for mastering the technical aspects and precision of long passes.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Define checkpoints based on ball travel distance
        # Define checkpoint rewards and distance thresholds for long passes.
        self.long_pass_thresholds = np.linspace(0.2, 0.9, 5)  # Progressive thresholds for longer passes
        self.long_pass_rewards = np.linspace(0.1, 0.5, 5)     # Progressive rewards for each threshold

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for agent_idx in range(len(reward)):
            # Extract relevant information for agent
            o = observation[agent_idx]
            ball_pos_last = o['ball']
            ball_direction = o['ball_direction']

            # We assume ball_direction gives the movement of the ball in the last step
            # Calculate the distance the ball traveled in the last action
            distance_traveled = np.linalg.norm(ball_direction[:2])  # Ignore z-axis

            # Check if the pass was long enough to get a reward
            for i, threshold in enumerate(self.long_pass_thresholds):
                if distance_traveled >= threshold:
                    # Reward only the best long pass achieved
                    components["long_pass_reward"][agent_idx] = max(components["long_pass_reward"][agent_idx],
                                                                    self.long_pass_rewards[i])
        
        # Update reward with additional long pass rewards
        reward = [base + bonus for base, bonus in zip(reward, components['long_pass_reward'])]

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
            for idx, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{idx}"] = action
        return observation, reward, done, info
