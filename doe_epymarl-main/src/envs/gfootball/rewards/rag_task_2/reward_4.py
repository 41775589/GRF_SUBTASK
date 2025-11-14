import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds dense rewards for dribbling, ball retention, and space creation."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._previous_ball_position = None
        self._previous_player_position = None
        self._ball_possession_steps = 0
        self._dribbling_reward_coef = 0.05
        self._retention_reward_coef = 0.02
        self._space_creation_reward_coef = 0.08
        self._attacking_third_bonus_coef = 2.0
        self._penalty_area_bonus_coef = 3.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = None
        self._previous_player_position = None
        self._ball_possession_steps = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self._previous_ball_position,
            'previous_player_position': self._previous_player_position,
            'ball_possession_steps': self._ball_possession_steps
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        checkpoint_state = from_pickle['CheckpointRewardWrapper']
        self._previous_ball_position = checkpoint_state['previous_ball_position']
        self._previous_player_position = checkpoint_state['previous_player_position']
        self._ball_possession_steps = checkpoint_state['ball_possession_steps']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0] * len(reward),
            "retention_reward": [0.0] * len(reward),
            "space_creation_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Start with base reward
            total_reward = reward[rew_index]
            
            # Check if our team (left team, index 0) has possession and active player has ball
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0 and
                'ball_owned_player' in o and o['ball_owned_player'] == o['active']):
                
                active_player_idx = o['active']
                player_pos = o['left_team'][active_player_idx]
                ball_pos = o['ball'][:2]  # Only x, y coordinates
                
                # Determine field zones for bonus multipliers
                attacking_third = ball_pos[0] > 0.33  # Right side of field
                penalty_area = ball_pos[0] > 0.83 and abs(ball_pos[1]) < 0.2  # Near opponent goal
                
                zone_multiplier = 1.0
                if penalty_area:
                    zone_multiplier = self._penalty_area_bonus_coef
                elif attacking_third:
                    zone_multiplier = self._attacking_third_bonus_coef
                
                # 1. Ball Retention Reward - reward for maintaining possession
                self._ball_possession_steps += 1
                retention_reward = self._retention_reward_coef * min(self._ball_possession_steps / 10.0, 1.0)
                components["retention_reward"][rew_index] = retention_reward * zone_multiplier
                total_reward += components["retention_reward"][rew_index]
                
                # 2. Dribbling Movement Reward - reward for moving ball while maintaining possession
                if self._previous_ball_position is not None and self._previous_player_position is not None:
                    # Calculate ball movement (indicates dribbling)
                    ball_movement = ((ball_pos[0] - self._previous_ball_position[0])**2 + 
                                   (ball_pos[1] - self._previous_ball_position[1])**2)**0.5
                    
                    # Calculate player movement
                    player_movement = ((player_pos[0] - self._previous_player_position[0])**2 + 
                                     (player_pos[1] - self._previous_player_position[1])**2)**0.5
                    
                    # Reward coordinated ball-player movement (dribbling)
                    if ball_movement > 0.001 and player_movement > 0.001:
                        # Check if ball and player are moving in similar direction (controlled dribbling)
                        ball_dir = [ball_pos[0] - self._previous_ball_position[0], 
                                   ball_pos[1] - self._previous_ball_position[1]]
                        player_dir = [player_pos[0] - self._previous_player_position[0],
                                     player_pos[1] - self._previous_player_position[1]]
                        
                        # Normalize and calculate similarity
                        if ball_movement > 0 and player_movement > 0:
                            ball_dir_norm = [ball_dir[0]/ball_movement, ball_dir[1]/ball_movement]
                            player_dir_norm = [player_dir[0]/player_movement, player_dir[1]/player_movement]
                            
                            # Dot product for direction similarity
                            direction_similarity = (ball_dir_norm[0] * player_dir_norm[0] + 
                                                  ball_dir_norm[1] * player_dir_norm[1])
                            
                            if direction_similarity > 0.5:  # Moving in similar direction
                                dribbling_reward = self._dribbling_reward_coef * ball_movement * 10
                                components["dribbling_reward"][rew_index] = dribbling_reward * zone_multiplier
                                total_reward += components["dribbling_reward"][rew_index]
                
                # 3. Space Creation Reward - reward for creating distance from opponents
                if len(o['right_team']) > 0:
                    # Find distance to nearest opponent
                    min_opponent_dist = float('inf')
                    for opponent_pos in o['right_team']:
                        dist = ((player_pos[0] - opponent_pos[0])**2 + 
                               (player_pos[1] - opponent_pos[1])**2)**0.5
                        min_opponent_dist = min(min_opponent_dist, dist)
                    
                    # Reward for maintaining/creating space from opponents while dribbling
                    if min_opponent_dist < 0.1:  # Very close to opponent
                        # High reward for maintaining possession under pressure
                        space_reward = self._space_creation_reward_coef * 2.0
                        components["space_creation_reward"][rew_index] = space_reward * zone_multiplier
                        total_reward += components["space_creation_reward"][rew_index]
                    elif min_opponent_dist > 0.05:  # Created some space
                        space_reward = self._space_creation_reward_coef * min_opponent_dist * 5
                        components["space_creation_reward"][rew_index] = space_reward * zone_multiplier
                        total_reward += components["space_creation_reward"][rew_index]
                
                # Update previous positions
                self._previous_ball_position = ball_pos.copy()
                self._previous_player_position = player_pos.copy()
                
            else:
                # Reset possession counter if we lose the ball
                self._ball_possession_steps = 0
                # Still update positions for continuity
                if 'ball' in o:
                    self._previous_ball_position = o['ball'][:2].copy()
                if 'left_team' in o and 'active' in o:
                    self._previous_player_position = o['left_team'][o['active']].copy()
            
            reward[rew_index] = total_reward

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
