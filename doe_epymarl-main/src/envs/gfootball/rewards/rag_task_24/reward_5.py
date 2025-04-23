import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that improves and rewards mid to long-range passes, emphasizing the precision and strategy in coordination.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for distance thresholds and rewards
        self.min_pass_distance = 0.2  # Minimum distance for a pass to be considered long-range
        self.max_pass_accuracy_threshold = 0.1  # Maximum acceptable deviation from targeted player
        self.pass_reward = 0.5  # Reward for successful long-range precise pass
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        # Initialize the components dictionary to store different components of the reward
        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": [0.0] * len(reward)
        }
        
        # Fetch current observation to check passing dynamics
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index, o in enumerate(observation):
            if o['game_mode'] != 0:
                continue  # Only modify reward during normal game play
            
            # Check if a pass has occurred
            if o['ball_owned_team'] == 1 and np.any(o['sticky_actions'][[0, 1, 2, 3, 4, 6, 7]]):
                ball_owner = o['ball_owned_player']
                ball_position = o['ball']
                team_position = o['right_team']
                
                if ball_owner != -1:
                    player_position = team_position[ball_owner]
                    for other_player_index, other_player_position in enumerate(team_position):
                        if other_player_index != ball_owner:
                            # Calculate distance between ball owner and other players
                            distance = np.linalg.norm(player_position - other_player_position)
                            
                            if distance >= self.min_pass_distance:
                                # Check if ball lands close to any teammate
                                ball_landing_position = player_position + o['ball_direction'][:2] * distance
                                landing_error = np.linalg.norm(ball_landing_position - other_player_position)
                                
                                if landing_error <= self.max_pass_accuracy_threshold:
                                    # Reward precise long-range passes
                                    components["pass_reward"][rew_index] += self.pass_reward
                                    reward[rew_index] += components["pass_reward"][rew_index]
        
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
