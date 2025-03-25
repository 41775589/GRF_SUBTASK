import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a goal-oriented skill reward based on offensive football skills."""

    def __init__(self, env):
        super().__init__(env)
        self.positional_rewards = [0.02, 0.02, 0.05, 0.01, 0.01]  # Specific rewards for [Short Pass, Long Pass, Shot, Dribble, Sprint]
        self.action_keys = ['action_short_pass', 'action_long_pass', 'action_shot', 'action_dribble', 'action_sprint']
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and reward tracking."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Currently no state used for reward."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Currently no state used for reward."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Calculate rewards based on the actions performed that relate to offensive football skills."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "skill_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            sticky_actions = o['sticky_actions']
            
            skill_reward = 0.0
            for j, action_key in enumerate(self.action_keys):
                action_index = football_action_set.action_set_to_index(action_key)
                if sticky_actions[action_index] == 1:
                    skill_reward += self.positional_rewards[j]
                    
            components["skill_reward"][i] = skill_reward
            reward[i] += skill_reward

        return reward, components

    def step(self, action):
        """Apply an action to the environment, augment reward, and return observation."""
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
