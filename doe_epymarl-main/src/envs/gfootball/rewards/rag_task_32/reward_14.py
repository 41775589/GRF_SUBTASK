import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper for enhancing training focused on winger crossing and sprinting capabilities."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions
    
    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Saves the current state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Sets the state from the saved state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Custom reward function focused on wingers crossing and sprinting."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "crossing_accuracy_reward": [0.0] * len(reward),
            "high_speed_dribbling_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # If our team owns the ball
                if 'active' in o and o['active'] in o['right_team_roles'] and o['right_team_roles'][o['active']] in [6, 7]:  # Check if the player is a winger
                    # Reward for crossing accuracy, considering the proximity to the goal line when crossing
                    if np.abs(o['ball'][1]) > 0.8 and o['ball'][0] > 0.5:
                        components["crossing_accuracy_reward"][rew_index] = 1.0  # Increase reward if ball is crossed near opponents' goal line

                    # Reward for dribbling at high speed
                    if self.sticky_actions_counter[8] > 0:  # 'action_sprint'
                        components["high_speed_dribbling_reward"][rew_index] += 0.1

            # Update rewards
            reward[rew_index] += components["crossing_accuracy_reward"][rew_index] + components["high_speed_dribbling_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        """Executes a step in the environment, applies the reward function, and gathers additional info."""
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
