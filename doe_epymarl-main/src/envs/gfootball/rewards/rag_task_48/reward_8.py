import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards high passes from midfield with the goal of creating direct scoring opportunities.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_area = [-0.5, 0.5]  # X-coordinates considered as midfield
        self.high_pass_height_threshold = 0.15  # threshold for z-coordinate to count as a high pass
        self.goal_area = [0.8, 1]  # X-coordinates close to the opposing goal
        self.high_pass_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """Recomputes the reward based on executing high passes from midfield.
        
        A favorable high pass is rewarded, especially if it leads towards a direct scoring opportunity.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball']
            ball_direction = o['ball_direction']
            ball_end_pos_x = ball_position[0] + ball_direction[0]
            
            # Check for high pass made from midfield
            if (self.midfield_area[0] <= ball_position[0] <= self.midfield_area[1] and
                ball_position[2] >= self.high_pass_height_threshold and
                self.goal_area[0] <= ball_end_pos_x <= self.goal_area[1]):
                
                components["high_pass_reward"][rew_index] = self.high_pass_reward
                reward[rew_index] += self.high_pass_reward
        
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
