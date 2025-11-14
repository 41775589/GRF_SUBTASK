import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards goal-mouth positioning and tap-in opportunities."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._dangerous_zone_time = {}  # Track time spent in dangerous zones
        self._last_ball_position = {}   # Track ball movement for rebound detection
        self._shots_taken = {}         # Track shots from dangerous positions
        self._goals_scored = {}        # Track goals from dangerous positions
        
        # Reward coefficients - can be adjusted for tuning
        self._dangerous_zone_reward = 0.01    # Per step in dangerous zone
        self._rebound_opportunity_reward = 0.1 # When ball is loose near goal
        self._close_shot_reward = 0.2         # For shots from close range
        self._tap_in_bonus = 0.5              # Extra reward for very close goals
        self._positioning_bonus = 0.05        # For optimal positioning
        
        # Zone definitions (closer to opponent goal = more dangerous)
        self._penalty_box_threshold = 0.75    # X coordinate for penalty area
        self._six_yard_box_threshold = 0.9    # X coordinate for six-yard box
        self._goal_mouth_threshold = 0.95     # Very close to goal
        self._goal_width = 0.044              # Half width of goal opening
        
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._dangerous_zone_time = {}
        self._last_ball_position = {}
        self._shots_taken = {}
        self._goals_scored = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'dangerous_zone_time': self._dangerous_zone_time,
            'last_ball_position': self._last_ball_position,
            'shots_taken': self._shots_taken,
            'goals_scored': self._goals_scored
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self._dangerous_zone_time = wrapper_state['dangerous_zone_time']
        self._last_ball_position = wrapper_state['last_ball_position']
        self._shots_taken = wrapper_state['shots_taken']
        self._goals_scored = wrapper_state['goals_scored']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dangerous_zone_reward": [0.0] * len(reward),
            "rebound_opportunity_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward),
            "tap_in_bonus_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Initialize tracking for this agent if needed
            if rew_index not in self._dangerous_zone_time:
                self._dangerous_zone_time[rew_index] = 0
                self._last_ball_position[rew_index] = None
                self._shots_taken[rew_index] = 0
                self._goals_scored[rew_index] = 0
            
            # Handle goal scoring with tap-in bonus
            if reward[rew_index] == 1:
                self._goals_scored[rew_index] += 1
                
                # Check if this was a tap-in (goal scored from very close range)
                active_player_pos = o['left_team'][o['active']]
                distance_to_goal = abs(1.0 - active_player_pos[0])  # Distance to right goal
                
                if distance_to_goal < (1.0 - self._goal_mouth_threshold):
                    components["tap_in_bonus_reward"][rew_index] = self._tap_in_bonus
                    reward[rew_index] += self._tap_in_bonus
                
                continue
            
            # Get active player position
            active_player_pos = o['left_team'][o['active']]
            ball_pos = o['ball']
            
            # Reward for being in dangerous zones near opponent goal
            distance_to_goal = abs(1.0 - active_player_pos[0])
            y_distance_from_center = abs(active_player_pos[1])
            
            # Check if player is in various dangerous zones
            in_penalty_box = active_player_pos[0] > self._penalty_box_threshold and y_distance_from_center < 0.25
            in_six_yard_box = active_player_pos[0] > self._six_yard_box_threshold and y_distance_from_center < 0.1
            in_goal_mouth = active_player_pos[0] > self._goal_mouth_threshold and y_distance_from_center < self._goal_width
            
            # Dangerous zone positioning rewards
            if in_goal_mouth:
                components["dangerous_zone_reward"][rew_index] = self._dangerous_zone_reward * 3
                self._dangerous_zone_time[rew_index] += 1
            elif in_six_yard_box:
                components["dangerous_zone_reward"][rew_index] = self._dangerous_zone_reward * 2
                self._dangerous_zone_time[rew_index] += 1
            elif in_penalty_box:
                components["dangerous_zone_reward"][rew_index] = self._dangerous_zone_reward
                self._dangerous_zone_time[rew_index] += 1
            
            # Detect rebound opportunities (ball is loose and moving in dangerous area)
            ball_in_dangerous_area = ball_pos[0] > self._penalty_box_threshold and abs(ball_pos[1]) < 0.25
            ball_is_loose = o['ball_owned_team'] == -1
            
            if ball_is_loose and ball_in_dangerous_area:
                # Check if ball is moving (potential rebound)
                if self._last_ball_position[rew_index] is not None:
                    ball_speed = ((ball_pos[0] - self._last_ball_position[rew_index][0])**2 + 
                                 (ball_pos[1] - self._last_ball_position[rew_index][1])**2)**0.5
                    
                    if ball_speed > 0.01:  # Ball is moving significantly
                        # Reward player for being close to loose ball in dangerous area
                        distance_to_ball = ((active_player_pos[0] - ball_pos[0])**2 + 
                                          (active_player_pos[1] - ball_pos[1])**2)**0.5
                        
                        if distance_to_ball < 0.05:  # Very close to loose ball
                            components["rebound_opportunity_reward"][rew_index] = self._rebound_opportunity_reward
            
            # Optimal positioning reward (being in good position relative to goal and ball)
            if ball_in_dangerous_area and not ball_is_loose:
                # Reward for being in optimal tap-in position
                optimal_x = min(ball_pos[0] + 0.05, 0.98)  # Slightly ahead of ball toward goal
                optimal_y = ball_pos[1] * 0.5  # Slightly toward center from ball
                
                distance_to_optimal = ((active_player_pos[0] - optimal_x)**2 + 
                                     (active_player_pos[1] - optimal_y)**2)**0.5
                
                if distance_to_optimal < 0.03:
                    components["positioning_reward"][rew_index] = self._positioning_bonus
            
            # Store ball position for next step
            self._last_ball_position[rew_index] = ball_pos.copy()
            
            # Apply all component rewards
            total_component_reward = (components["dangerous_zone_reward"][rew_index] + 
                                    components["rebound_opportunity_reward"][rew_index] + 
                                    components["positioning_reward"][rew_index] + 
                                    components["tap_in_bonus_reward"][rew_index])
            
            reward[rew_index] += total_component_reward
        
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
