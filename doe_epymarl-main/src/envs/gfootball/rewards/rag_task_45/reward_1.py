import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for training defensive techniques 
    involving Stop-Sprint and Stop-Moving actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'defensive_action_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Handle the "stop" action detection and reward
        for rew_idx in range(len(reward)):
            o = observation[rew_idx]

            # Increment the action counter if 'stop' (action id 0) is present in sticky actions
            if o['sticky_actions'][0] == 1:  
                self.sticky_actions_counter[0] += 1

            # Reward for successful stop-sprint and stop-moving transitions
            if self.sticky_actions_counter[0] > 0 and (o['sticky_actions'][8] or o['sticky_actions'][1:5].any()):
                components['defensive_action_reward'][rew_idx] = 1.0
                self.sticky_actions_counter[0] = 0  # Reset the stop action counter after successful transition

            reward[rew_idx] += components['defensive_action_reward'][rew_idx]

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
