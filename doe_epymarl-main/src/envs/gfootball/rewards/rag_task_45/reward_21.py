import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on teaching agents to stop and change direction
    quickly, simulating defensive maneuvers in football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Store the action history to detect stopping and direction changes
        self.action_history = []
        self.direction_change_bonus = 0.2
        self.stop_bonus = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        # Resetting the history on environment reset
        self.action_history = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'action_history': self.action_history}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.action_history = from_pickle['CheckpointRewardWrapper']['action_history']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_bonus_reward": [0.0] * len(reward),
                      "direction_change_bonus_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for agent_idx in range(len(reward)):
            o = observation[agent_idx]

            # Detect stopping: abrupt absence of movement while in motion previously
            if self.action_history and np.linalg.norm(o['left_team_direction'][o['active']]) == 0:
                if np.linalg.norm(self.action_history[-1]['left_team_direction'][self.action_history[-1]['active']]) > 0:
                    reward[agent_idx] += self.stop_bonus
                    components["stop_bonus_reward"][agent_idx] += self.stop_bonus
            
            # Detect direction change: detect shifts in movement vector direction
            if self.action_history and np.dot(o['left_team_direction'][o['active']], 
                                             self.action_history[-1]['left_team_direction'][self.action_history[-1]['active']]) < 0:
                reward[agent_idx] += self.direction_change_bonus
                components["direction_change_bonus_reward"][agent_idx] += self.direction_change_bonus
            
            self.action_history.append(o)
        
        return reward, components

    def step(self, action):
        # Execute step in environment
        obs, reward, done, info = self.env.step(action)
        # Apply reward wrapper mechanics
        reward, components = self.reward(reward)
        # Store the rewards in info for debugging 
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Reset history if episode ended
        if done:
            self.action_history.clear()
        return obs, reward, done, info
