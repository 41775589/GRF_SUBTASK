import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense dynamic transition reward, focusing on stopping and starting movements."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward thresholds and initial state
        self.threshold_speed_change = 0.1  # Arbitrary small threshold value for significant speed change
        self.prev_speed = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_speed = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['prev_speed'] = self.prev_speed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.prev_speed = from_pickle['prev_speed']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward).copy(),
                      "transition_reward": np.zeros(len(reward))}
        
        if not observation:
            return reward, components
        
        for idx, obs in enumerate(observation):
            player_speed = np.sqrt(obs['left_team_direction'][obs['active'], 0]**2 + 
                                   obs['left_team_direction'][obs['active'], 1]**2)
            if self.prev_speed is not None:
                speed_change = abs(player_speed - self.prev_speed[idx])
                # Reward is given based on the change from moving to stopping and vice versa
                if speed_change > self.threshold_speed_change:
                    components["transition_reward"][idx] = 0.5  # Assign transition reward
            reward[idx] += components["transition_reward"][idx]
        
        self.prev_speed = [np.sqrt(obs['left_team_direction'][obs['active'], 0]**2 +
                                   obs['left_team_direction'][obs['active'], 1]**2)
                                for obs in observation]  # Update previous speed state
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
            for i, action_ in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_  # Tally all sticky actions
        
        for i, count in enumerate(self.sticky_actions_counter):
            info[f"sticky_actions_{i}"] = count
                
        return observation, reward, done, info
