import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that customizes the reward to incentivize mid to long-range passing effectiveness.
    The reward is increased when passes are correctly executed over longer distances and when
    they contribute to strategic moves toward the opponent's goal.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # Define a range which is considered mid to long-range
        self.mid_to_long_range_threshold = 0.2  # This threshold is set considering the field length
        self.passing_bonus = 0.2  # Reward bonus for successful long passes
        self.strategic_movement_bonus = 0.1  # Bonus for movement towards opponent's goal
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_bonus": [0.0] * len(reward),
                      "strategic_movement_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            if not (o['ball_owned_team'] in (0, 1)):
                continue  # Only reward when the ball is not lost
            
            ball_pos_before = np.array(o['ball'])
            ball_owner = o['ball_owned_player']
            team = 'right_team' if o['ball_owned_team'] == 1 else 'left_team'
            team_directions = team + '_direction'
            team_members_pos = np.array(o[team])
            
            if ball_owner >= 0 and team_members_pos.size > ball_owner:  # Validate ball owner index
                player_pos = team_members_pos[ball_owner]
                player_to_ball_vector = ball_pos_before - player_pos
                pass_length = np.linalg.norm(player_to_ball_vector)
                
                # Check if this is a mid or long-range pass
                if pass_length > self.mid_to_long_range_threshold:
                    components['passing_bonus'][rew_index] = self.passing_bonus
                    reward[rew_index] += components['passing_bonus'][rew_index]
                
                # Check movement towards the opponent's goal
                if o['ball_owned_team'] == 0:
                    # Assuming moving right towards +1 is towards opponent's goal (ball owned by left team)
                    if player_to_ball_vector[0] > 0:
                        components['strategic_movement_bonus'][rew_index] = self.strategic_movement_bonus
                elif o['ball_owned_team'] == 1:
                    # Moving left towards -1 (ball owned by the right team)
                    if player_to_ball_vector[0] < 0:
                        components['strategic_movement_bonus'][rew_index] = self.strategic_movement_bonus
                
                reward[rew_index] += components['strategic_movement_bonus'][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Update the final reward and add breakdown to info
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Record the sticky actions from observation
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                # This key format matches the requirement of only counting action activations
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
