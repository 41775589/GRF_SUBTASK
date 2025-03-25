import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds dense rewards based on progression to the opponent's goal and specific player actions indicating fast-paced maneuvering and precision play.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Defines zones on the field for checkpoint rewards
        self.checkpoints = np.linspace(-1, 1, 8)
        self.checkpoints_collected = set()
        self.checkpoint_reward = 0.2
        self.precision_control_reward = 0.05
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoints_collected = set()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_count'] = self.sticky_actions_counter
        to_pickle['CheckpointRewardWrapper_checkpoints'] = self.checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_count']
        self.checkpoints_collected = from_pickle['CheckpointRewardWrapper_checkpoints']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "checkpoint_reward": [0.0] * len(reward),
                      "precision_control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            initial_reward = reward[rew_index]

            # Checkpoint logic
            x_position = o['ball'][0]
            for checkpoint in self.checkpoints:
                if x_position >= checkpoint and checkpoint not in self.checkpoints_collected:
                    self.checkpoints_collected.add(checkpoint)
                    components["checkpoint_reward"][rew_index] += self.checkpoint_reward
                    initial_reward += self.checkpoint_reward
            
            # Precision and control based rewards
            if o['sticky_actions'][8] == 1:  # Sprint active
                components["precision_control_reward"][rew_index] += self.precision_control_reward
                initial_reward += self.precision_control_reward
            if o['sticky_actions'][9] == 1:  # Dribble active
                components["precision_control_reward"][rew_index] += self.precision_control_reward
                initial_reward += self.precision_control_reward

            reward[rew_index] = initial_reward

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
