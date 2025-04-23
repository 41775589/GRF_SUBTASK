import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering close-range attacks
    via shot precision and dribble effectiveness against goalkeepers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.max_distance_reward = 0.2
        self.close_range_threshold = 0.1
        self.dribble_reward_multiplier = 0.03
        self.precision_reward_multiplier = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Store original reward for output
        base_reward = reward.copy()
        
        # Get current observation
        observation = self.env.unwrapped.observation()
        
        # Initialize reward components
        components = {
            "base_score_reward": base_reward,
            "close_range_attack_reward": [0.0] * len(reward),
            "dribble_effectiveness_reward": [0.0] * len(reward),
        }
        
        # Apply close-range attack and dribble effectiveness rewards
        for idx, o in enumerate(observation):
            distance_to_goal = ((1 - o['ball'][0])**2 + o['ball'][1]**2)**0.5
            if distance_to_goal < self.close_range_threshold:
                components["close_range_attack_reward"][idx] = self.max_distance_reward
            if o['sticky_actions'][9]:  # Assuming '9' is the dribble action
                components["dribble_effectiveness_reward"][idx] = self.dribble_reward_multiplier * (1 - distance_to_goal)
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                components["close_range_attack_reward"][idx] += self.precision_reward_multiplier * (1 - distance_to_goal)
            
            # Encourage quick, effective shots at close range
            reward[idx] += (components["close_range_attack_reward"][idx] + 
                            components["dribble_effectiveness_reward"][idx])
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions stats for observation
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
