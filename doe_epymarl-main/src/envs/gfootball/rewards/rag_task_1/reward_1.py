import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for long-range shooting behavior."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._long_range_shot_reward = 0.3
        self._medium_range_shot_reward = 0.2
        self._shot_quality_reward = 0.15
        self._goal_bonus_multiplier = 2.0
        self._ball_possession_reward = 0.05
        self._previous_ball_position = {}
        self._previous_ball_owned_team = {}
        self._previous_ball_owned_player = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = {}
        self._previous_ball_owned_team = {}
        self._previous_ball_owned_player = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self._previous_ball_position,
            'previous_ball_owned_team': self._previous_ball_owned_team,
            'previous_ball_owned_player': self._previous_ball_owned_player
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle.get('CheckpointRewardWrapper', {})
        self._previous_ball_position = wrapper_state.get('previous_ball_position', {})
        self._previous_ball_owned_team = wrapper_state.get('previous_ball_owned_team', {})
        self._previous_ball_owned_player = wrapper_state.get('previous_ball_owned_player', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_range_shot_reward": [0.0] * len(reward),
            "medium_range_shot_reward": [0.0] * len(reward),
            "shot_quality_reward": [0.0] * len(reward),
            "ball_possession_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # If agent scored a goal, apply bonus multiplier to shooting rewards
            if reward[rew_index] == 1:
                # Keep the original goal reward and add bonus for long-range goals
                reward[rew_index] = reward[rew_index] * self._goal_bonus_multiplier
                continue

            # Get current ball and player information
            ball_pos = o.get('ball', [0, 0, 0])
            ball_owned_team = o.get('ball_owned_team', -1)
            ball_owned_player = o.get('ball_owned_player', -1)
            active_player = o.get('active', 0)
            left_team = o.get('left_team', [])
            
            # Check if our team (left team = 0) has the ball
            if ball_owned_team == 0 and len(left_team) > active_player:
                player_pos = left_team[active_player]
                
                # Calculate distance to opponent's goal (right goal at x=1)
                goal_distance = ((1 - player_pos[0]) ** 2 + (0 - player_pos[1]) ** 2) ** 0.5
                
                # Reward for maintaining ball possession in attacking positions
                if player_pos[0] > 0:  # In opponent's half
                    components["ball_possession_reward"][rew_index] = self._ball_possession_reward
                    reward[rew_index] += components["ball_possession_reward"][rew_index]

            # Detect potential shots by analyzing ball movement
            prev_ball_pos = self._previous_ball_position.get(rew_index, ball_pos)
            prev_owned_team = self._previous_ball_owned_team.get(rew_index, -1)
            prev_owned_player = self._previous_ball_owned_player.get(rew_index, -1)
            
            # Check if ball was owned by our team and is now moving towards goal
            if (prev_owned_team == 0 and ball_owned_team == -1 and 
                len(left_team) > 0):
                
                # Ball is no longer owned and is moving - potential shot
                ball_direction = o.get('ball_direction', [0, 0, 0])
                
                # Check if ball is moving towards opponent's goal (positive x direction)
                if ball_direction[0] > 0.01:
                    # Determine shot distance based on previous ball position
                    shot_distance = ((1 - prev_ball_pos[0]) ** 2 + (0 - prev_ball_pos[1]) ** 2) ** 0.5
                    
                    # Long-range shot (distance > 0.4, roughly penalty arc and beyond)
                    if shot_distance > 0.4:
                        components["long_range_shot_reward"][rew_index] = self._long_range_shot_reward
                        reward[rew_index] += components["long_range_shot_reward"][rew_index]
                        
                        # Additional reward for shot quality (good angle towards goal)
                        goal_center = [1, 0]
                        shot_angle_quality = self._calculate_shot_quality(prev_ball_pos, ball_direction, goal_center)
                        if shot_angle_quality > 0.5:
                            components["shot_quality_reward"][rew_index] = self._shot_quality_reward
                            reward[rew_index] += components["shot_quality_reward"][rew_index]
                    
                    # Medium-range shot (distance 0.2 to 0.4)
                    elif shot_distance > 0.2:
                        components["medium_range_shot_reward"][rew_index] = self._medium_range_shot_reward
                        reward[rew_index] += components["medium_range_shot_reward"][rew_index]
                        
                        # Quality bonus for medium range shots too
                        goal_center = [1, 0]
                        shot_angle_quality = self._calculate_shot_quality(prev_ball_pos, ball_direction, goal_center)
                        if shot_angle_quality > 0.5:
                            components["shot_quality_reward"][rew_index] = self._shot_quality_reward * 0.5
                            reward[rew_index] += components["shot_quality_reward"][rew_index]

            # Update previous state
            self._previous_ball_position[rew_index] = ball_pos.copy()
            self._previous_ball_owned_team[rew_index] = ball_owned_team
            self._previous_ball_owned_player[rew_index] = ball_owned_player

        return reward, components
    
    def _calculate_shot_quality(self, ball_pos, ball_direction, goal_center):
        """Calculate shot quality based on ball direction towards goal center."""
        # Vector from ball to goal center
        to_goal = [goal_center[0] - ball_pos[0], goal_center[1] - ball_pos[1]]
        to_goal_magnitude = (to_goal[0] ** 2 + to_goal[1] ** 2) ** 0.5
        
        if to_goal_magnitude == 0:
            return 0
        
        # Normalize vectors
        to_goal_norm = [to_goal[0] / to_goal_magnitude, to_goal[1] / to_goal_magnitude]
        ball_dir_magnitude = (ball_direction[0] ** 2 + ball_direction[1] ** 2) ** 0.5
        
        if ball_dir_magnitude == 0:
            return 0
        
        ball_dir_norm = [ball_direction[0] / ball_dir_magnitude, ball_direction[1] / ball_dir_magnitude]
        
        # Dot product to measure alignment (cosine similarity)
        alignment = to_goal_norm[0] * ball_dir_norm[0] + to_goal_norm[1] * ball_dir_norm[1]
        
        # Return quality score (0 to 1, where 1 is perfect alignment)
        return max(0, alignment)

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
