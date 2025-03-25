import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive actions and positioning to prevent opponent scoring."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Specific defensive actions we want to reward
        self.defensive_actions_reward = {
            'slide': 0.5,  # action index for sliding tackle, value is a reward scaler
            'stop_dribble': 0.3, # action index for stopping a dribble, value is a reward scaler
            'stop_moving': 0.2  # action index for stopping movement, value is a reward scaler
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
                      "defensive_action_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            action_rewards = 0
            # Reward defensive actions
            for action_idx, action_value in self.defensive_actions_reward.items():
                if o['sticky_actions'][action_idx]:
                    action_rewards += action_value
                    self.sticky_actions_counter[action_idx] += 1
            # Adjust the reward based on the action taken and the scenario
            reward[rew_index] += action_rewards
            components["defensive_action_reward"][rew_index] = action_rewards
        
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
