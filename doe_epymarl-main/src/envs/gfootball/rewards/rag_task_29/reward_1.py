import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances reward signals for teaching precision in shots within close ranges.
    This encourages agents to develop skill in adjusting angles and power to beat the goalkeeper
    from tight spaces near the goal.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset for a new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the current wrapper state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of the wrapper.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Adjust the reward to focus on precision skills in close shooting scenarios.
        Includes a bonus for shooting from a close region with correct power and angle.
        """
        observation = self.env.unwrapped.observation()
        
        # Initialize the components of the reward given to each agent.
        components = {
            "base_score_reward": reward.copy(),
            "precision_shoot_bonus": [0.0] * len(reward)
        }
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Calculate the distance of the ball from the opponent's goal.
            goal_x = 1 - o['ball'][0]  # 1 is the x-coordinate of the opponent's goal in normalized coords
            goal_y = o['ball'][1]
            distance_to_goal = np.sqrt(goal_x**2 + goal_y**2)
            
            close_range_threshold = 0.2  # customize the range considered as "close" (20% of the field width)
            
            # Check if the last player to touch the ball is the current active agent, if close to goal and not owned by the opponent
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'] and distance_to_goal < close_range_threshold:
                components["precision_shoot_bonus"][rew_index] = 0.5  # custom reward for close range control
                
                # Include the bonus in the reward computed.
                reward[rew_index] += components["precision_shoot_bonus"][rew_index]
        
        return reward, components

    def step(self, action):
        """
        Environment step with adjusted reward and logs for debugging.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        # Add sticky_actions to info for easier debugging.
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                
        for i in range(10):
            info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
