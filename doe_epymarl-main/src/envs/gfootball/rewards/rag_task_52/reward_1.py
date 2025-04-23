import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on enhancing defending strategies by rewarding tackling proficiency,
    efficient movement control, and pressured passing tactics."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Parameters for reward customization
        self.tackle_reward = 0.5
        self.movement_efficiency_reward = 0.3
        self.pressured_pass_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the sticky actions tracker."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serializes the current state of the wrapper along with the environment state."""
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Resets the state from deserialized state and environment's state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Calculates additional rewards for defending capabilities based on the current observation of the game."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "movement_efficiency_reward": [0.0] * len(reward),
                      "pressured_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Tackle reward: modify base reward if the ball is tackled effectively
            if o['ball_owned_team'] == 1 and o['sticky_actions'][9]:  # Consider dribble action as a tactical move
                components["tackle_reward"][rew_index] += self.tackle_reward
                reward[rew_index] += components["tackle_reward"][rew_index]
            
            # Movement efficiency: encourage minimal unnecessary movement when defending
            if o['sticky_actions'][0] or o['sticky_actions'][4]:  # left or right movement
                components["movement_efficiency_reward"][rew_index] -= self.movement_efficiency_reward
                reward[rew_index] += components["movement_efficiency_reward"][rew_index]
            
            # Pressured passing: reward for maintaining possession under pressure
            if o['ball_owned_team'] == 0 and any(o['sticky_actions'][:8]):  # active passing under pressure
                components["pressured_pass_reward"][rew_index] += self.pressured_pass_reward
                reward[rew_index] += components["pressured_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        """ Steps through environment, modifies rewards, and returns observations and info dictionary with reward components."""
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
