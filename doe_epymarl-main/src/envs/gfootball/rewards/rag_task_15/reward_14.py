import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards based on the accuracy and distance of long passes
    in the Google Research Football environment. This encourages mastering the 
    technical and precision aspects of long passes under varying match conditions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Define thresholds for long pass detection
        self.long_pass_min_distance = 0.3  # Minimum distance a ball must travel to be considered a long pass
        self.accuracy_threshold = 0.1      # Maximum distance from a teammate to consider the pass successful

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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Rewards are only computed when the ball is owned by the team
            if o['ball_owned_team'] != o['team']:
                continue
            
            ball_pos_before = o['ball']
            ball_dir = o['ball_direction']
            ball_pos_after = ball_pos_before + ball_dir
            ball_travel_distance = np.linalg.norm(ball_dir[:2])
            
            # Check if pass is a long pass
            if ball_travel_distance >= self.long_pass_min_distance:
                # Determine closest teammate distance after pass
                own_team = o['left_team'] if o['team'] == 0 else o['right_team']
                distances = [
                    np.linalg.norm(ball_pos_after[:2] - player_pos[:2])
                    for player_pos in own_team
                ]
                min_distance = min(distances) if distances else float('inf')
                
                # If pass is close to any teammate, consider it accurate
                if min_distance <= self.accuracy_threshold:
                    components["long_pass_reward"][rew_index] = 1.0
                    reward[rew_index] += components["long_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        # Apply the custom reward modifications
        reward, components = self.reward(reward)
        
        # Accumulate final reward and breakdown in info dictionary for analysis
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions for transparency in debugging
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action 
        
        # Return the processed results
        return observation, reward, done, info
