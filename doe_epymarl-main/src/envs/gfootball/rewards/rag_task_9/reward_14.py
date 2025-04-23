import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards offensive skills like passing, shooting, and dribbling."""
    
    def __init__(self, env):
        super().__init__(env)
        # Initialize parameters for different actions
        self.short_pass_reward = 0.05
        self.long_pass_reward = 0.1
        self.shot_reward = 0.2
        self.dribble_reward = 0.03  # Dribble should encourage maintaining control
        self.sprint_reward = 0.01  # Sprinting is useful but shouldn't be overemphasized
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset reward tracking
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save the state of the environment
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore the state of the environemnt from previous checkpoints
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Get the current observation to modify the reward based on actions taken
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "short_pass_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o.get('sticky_actions', [])

            # Apply rewards based on specific sticky actions
            if sticky_actions[0] or sticky_actions[1]:  # 0: short pass, 1: long pass
                components["short_pass_reward"][rew_index] = self.short_pass_reward
                components["long_pass_reward"][rew_index] = self.long_pass_reward
            if sticky_actions[6]:  # 6: shot
                components["shot_reward"][rew_index] = self.shot_reward
            if sticky_actions[9]:  # 9: dribble
                components["dribble_reward"][rew_index] = self.dribble_reward
            if sticky_actions[8]:  # 8: sprint
                components["sprint_reward"][rew_index] = self.sprint_reward

            # Summing up the additional rewards
            reward[rew_index] += (
                components["short_pass_reward"][rew_index] +
                components["long_pass_reward"][rew_index] +
                components["shot_reward"][rew_index] +
                components["dribble_reward"][rew_index] +
                components["sprint_reward"][rew_index]
            )
        
        return reward, components

    def step(self, action):
        # Execute environment step
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
