import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a task-specific reward focused on dribbling and dynamic positioning."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_status = [False] * 2  # Two players: [0] for left, [1] for right

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_status = [False] * 2
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_rewards": [0.0, 0.0],
                      "dynamic_positioning_rewards": [0.0, 0.0]}
        
        for agent_idx, agent_reward in enumerate(reward):
            o = observation[agent_idx]
            
            # Reward for dribbling initiations and stops
            if o['sticky_actions'][9] == 1 and not self.dribble_status[agent_idx]:  # Dribble started
                components["dribbling_rewards"][agent_idx] = 0.05
                self.dribble_status[agent_idx] = True
            elif o['sticky_actions'][9] == 0 and self.dribble_status[agent_idx]:  # Dribble stopped
                components["dribbling_rewards"][agent_idx] = 0.05
                self.dribble_status[agent_idx] = False
            
            # Reward for dynamic positioning: Encourages changing x position effectively
            if abs(o['right_team'][o['active']][0] - o['left_team'][o['active']][0]) > 0.1:
                components["dynamic_positioning_rewards"][agent_idx] = 0.02
            
            agent_reward += components["dribbling_rewards"][agent_idx]
            agent_reward += components["dynamic_positioning_rewards"][agent_idx]
            reward[agent_idx] = agent_reward
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action != 0:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
