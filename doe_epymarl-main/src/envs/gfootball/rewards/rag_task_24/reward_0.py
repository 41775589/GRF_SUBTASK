import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for effective mid to long-range passing, 
    encouraging precision and strategic use in coordinated plays.
    """
    def __init__(self, env):
        super().__init__(env)
        self.passing_distance_threshold = 0.2  # Define threshold for considering a pass as mid to long-range
        self.pass_precision_reward = 0.05  # Reward for precise pass within the range
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset sticky actions counter
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore any necessary state
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_precision_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o:
                if o['ball_owned_team'] == 1:  # Team right owns the ball
                    current_player = o['active']
                    prior_pos = o['right_team'][current_player]
                    current_pos = o['right_team_direction'][current_player] + prior_pos
                    distance = np.linalg.norm(current_pos - prior_pos)
                    # Check if it's a long-range pass and successful control
                    if distance > self.passing_distance_threshold:
                        components["pass_precision_reward"][rew_index] = self.pass_precision_reward
                        reward[rew_index] += components["pass_precision_reward"][rew_index]
        
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
