import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes precise high passes through specified checkpoints in the 3D space of the game.
    It rewards agents for passing the ball through vertical 'gates' preserving height and precision, which is crucial
    for mastering high and long passes in football.
    """
    def __init__(self, env):
        super().__init__(env)
        self.checkpoint_heights = np.linspace(0.2, 0.5, 5)  # Heights for passing checkpoints
        self.passing_threshold = 0.05  # Threshold distance to consider a pass through the checkpoint
        self.checkpoint_depth = 0.5  # Zone after player's own field to aim the pass checkpoints.
        self.checkpoint_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            agent_obs = observation[rew_index]
            if agent_obs['ball'][2] > min(self.checkpoint_heights) and agent_obs['ball'][2] < max(self.checkpoint_heights):
                # Check if the ball is in the vertical checkpoint zone
                ball_height_index = np.digitize(agent_obs['ball'][2], self.checkpoint_heights) - 1
                target_height = self.checkpoint_heights[ball_height_index]
                
                if abs(agent_obs['ball'][1]) < self.passing_threshold and \
                   agent_obs['ball'][0] > -self.checkpoint_depth and \
                   agent_obs['ball'][0] < self.checkpoint_depth:
                    # Ball is inside one of the height checkpoints
                    key = (rew_index, ball_height_index)
                    if not self._collected_checkpoints.get(key, False):
                        # Reward for passing through this checkpoint
                        components["checkpoint_reward"][rew_index] += self.checkpoint_reward
                        reward[rew_index] += components["checkpoint_reward"][rew_index]
                        self._collected_checkpoints[key] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions info
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
