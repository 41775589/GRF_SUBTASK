import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper tailored for promoting energy conservation through
    proficient use of Stop-Sprint and Stop-Moving actions, critical
    for maintaining stamina and positional integrity over the match duration."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.action_usage_rewards = {
            8: -0.1,  # action_sprint
            4: -0.1,  # action_right e.g., stop moving right
            0: -0.1,  # action_left e.g., stop moving left
            2: -0.1,  # action_top e.g., stop moving up
            6: -0.1   # action_bottom e.g., stop moving down
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "action_usage_penalty": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sticky_actions = o.get('sticky_actions', [])
            
            # Apply penalties for using sprint or stop moving unnecessarily
            for index, action_status in enumerate(sticky_actions):
                if action_status and index in self.action_usage_rewards:
                    components["action_usage_penalty"][rew_index] += self.action_usage_rewards[index]
        
        # Update reward based on penalties
        reward = [rew + pen for rew, pen in zip(reward, components["action_usage_penalty"])]

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
