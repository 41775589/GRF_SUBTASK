import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized goalkeeper reward function."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_support_reward = 0.1
        self.max_save_distance_threshold = 0.3  # within 30% distance from goal
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Currently, we do not load any specific states for this wrapper.
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_support_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
            
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_role = np.argmin(o['left_team_roles'])  # Goalkeeper presumed to be first index

            if 'ball_owned_player' in o and o['ball_owned_player'] == player_role and o['ball_owned_team'] == 0:
                ball_pos = o['ball']
                goal_pos = [-1, 0]  # Assuming a standard goal position at left side
                distance_to_goal = np.linalg.norm(np.array(goal_pos) - np.array(ball_pos[:2]))
                
                if distance_to_goal < self.max_save_distance_threshold:
                    components["goalkeeper_support_reward"][rew_index] = self.goalkeeper_support_reward
                    reward[rew_index] += components["goalkeeper_support_reward"][rew_index]

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
