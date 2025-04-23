import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized rewards for mastering Stop-Sprint and Stop-Moving techniques."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track usage of sticky actions
        # Setting thresholds for what counts as 'stopping', 'sprinting', and 'moving'
        self.stop_threshold = 0.01  # Almost no movement is considered stopping
        self.sprint_threshold = 0.07  # Significant movement considered as sprinting
        self.coefficients = {
            "stop_reward": 0.1,
            "sprint_reward": 0.15,
            "move_penalty": -0.05
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return state

    def set_state(self, state):
        self.env.set_state(state)
        from_pickle = state['CheckpointRewardWrapper']
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return state

    def reward(self, reward):
        """Adjust the reward based on the agent's ability to stop and sprint effectively."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": list(reward),
            "stop_reward": [0.0]*len(reward),
            "sprint_reward": [0.0]*len(reward),
            "move_penalty": [0.0]*len(reward)
        }
        
        for idx, o in enumerate(observation):
            speed = np.linalg.norm(o['right_team_direction'][o['active']][:2])  # Calculate speed of the controlled player
            
            if speed < self.stop_threshold:
                # Reward for stopping
                components["stop_reward"][idx] = self.coefficients["stop_reward"]
            elif speed > self.sprint_threshold:
                # Reward for sprinting
                components["sprint_reward"][idx] = self.coefficients["sprint_reward"]
            else:
                # Penalty for moving but not sprinting
                components["move_penalty"][idx] = self.coefficients["move_penalty"]

            # Summing all components to form the final reward
            reward[idx] = reward[idx] + components["stop_reward"][idx] + components["sprint_reward"][idx] + components["move_penalty"][idx]
        
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
