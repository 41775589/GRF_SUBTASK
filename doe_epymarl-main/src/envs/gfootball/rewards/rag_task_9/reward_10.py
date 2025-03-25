import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for key football actions relevant to offensive skills.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.action_rewards = {
            'short_pass': 0.05,
            'long_pass': 0.1,
            'shot': 0.5,
            'dribble': 0.04,
            'sprint': 0.02
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
        components = {'base_score_reward': reward.copy(), 'action_rewards': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Determine the active action of the controlled player
            active_actions = o['sticky_actions']
            sticky_action_names = ['action_left', 'action_top_left', 'action_top', 
                                  'action_top_right', 'action_right', 'action_bottom_right', 
                                  'action_bottom', 'action_bottom_left', 'action_sprint', 
                                  'action_dribble']
            for i, action_value in enumerate(active_actions):
                if action_value == 1:
                    action_name = sticky_action_names[i]
                    if action_name in self.action_rewards:
                        components['action_rewards'][rew_index] += self.action_rewards[action_name]
                        reward[rew_index] += self.action_rewards[action_name]

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
