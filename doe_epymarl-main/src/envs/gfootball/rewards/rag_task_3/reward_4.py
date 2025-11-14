import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards goal-mouth positioning and tap-in opportunities."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward coefficients for easy tuning
        self._positioning_reward_coeff = 0.01  # Small continuous reward for good positioning
        self._opportunity_reward_coeff = 0.3   # Reward for being near ball in dangerous area
        self._tap_in_bonus_coeff = 0.5         # Additional reward for close-range goals
        self._shot_attempt_coeff = 0.1         # Reward for attempting shots in danger zone
        
        # Zone definitions (opponent's penalty area and near-goal regions)
        self._danger_zone_x_min = 0.6          # Danger zone starts at x=0.6 (opponent's half)
        self._danger_zone_x_close = 0.85       # Very close to goal for tap-ins
        self._danger_zone_y_range = 0.2        # Y range around goal center
        
        # State tracking
        self._previous_ball_owned_team = {}
        self._previous_ball_position = {}
        self._agent_positioning_time = {}
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_owned_team = {}
        self._previous_ball_position = {}
        self._agent_positioning_time = {}
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_owned_team': self._previous_ball_owned_team,
            'previous_ball_position': self._previous_ball_position,
            'agent_positioning_time': self._agent_positioning_time
        }
        return self.env.get_state(to_pickle)
        
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle.get('CheckpointRewardWrapper', {})
        self._previous_ball_owned_team = wrapper_state.get('previous_ball_owned_team', {})
        self._previous_ball_position = wrapper_state.get('previous_ball_position', {})
        self._agent_positioning_time = wrapper_state.get('agent_positioning_time', {})
        return from_pickle
        
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0] * len(reward),
            "opportunity_reward": [0.0] * len(reward),
            "tap_in_bonus": [0.0] * len(reward),
            "shot_attempt_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
            
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Handle goal scoring with potential tap-in bonus
            if reward[rew_index] == 1:  # Goal scored
                ball_pos = o.get('ball', [0, 0, 0])
                # If goal was scored from very close range (tap-in)
                if ball_pos[0] > self._danger_zone_x_close and abs(ball_pos[1]) < self._danger_zone_y_range:
                    components["tap_in_bonus"][rew_index] = self._tap_in_bonus_coeff
                    
                reward[rew_index] = (components["base_score_reward"][rew_index] + 
                                   components["tap_in_bonus"][rew_index])
                continue
                
            # Get current game state
            active_player_idx = o.get('active', 0)
            left_team_positions = o.get('left_team', np.array([[0, 0]]))
            ball_pos = o.get('ball', [0, 0, 0])
            ball_owned_team = o.get('ball_owned_team', -1)
            ball_owned_player = o.get('ball_owned_player', -1)
            
            # Get active player position
            if active_player_idx < len(left_team_positions):
                player_pos = left_team_positions[active_player_idx]
                
                # 1. Positioning reward - reward for being in dangerous areas
                in_danger_zone = (player_pos[0] > self._danger_zone_x_min and 
                                abs(player_pos[1]) < self._danger_zone_y_range * 1.5)
                                
                in_close_danger_zone = (player_pos[0] > self._danger_zone_x_close and 
                                      abs(player_pos[1]) < self._danger_zone_y_range)
                
                if in_close_danger_zone:
                    components["positioning_reward"][rew_index] = self._positioning_reward_coeff * 2.0
                    self._agent_positioning_time[rew_index] = self._agent_positioning_time.get(rew_index, 0) + 1
                elif in_danger_zone:
                    components["positioning_reward"][rew_index] = self._positioning_reward_coeff
                    self._agent_positioning_time[rew_index] = self._agent_positioning_time.get(rew_index, 0) + 1
                    
                # 2. Opportunity reward - reward for being near loose balls or rebounds in danger zone
                ball_distance = np.linalg.norm([ball_pos[0] - player_pos[0], ball_pos[1] - player_pos[1]])
                ball_in_danger_zone = (ball_pos[0] > self._danger_zone_x_min and 
                                     abs(ball_pos[1]) < self._danger_zone_y_range * 1.2)
                
                # Check for opportunity scenarios
                if ball_in_danger_zone and ball_distance < 0.1:  # Very close to ball in danger zone
                    # Loose ball scenario (great opportunity for tap-in)
                    if ball_owned_team == -1:
                        components["opportunity_reward"][rew_index] = self._opportunity_reward_coeff * 1.5
                    # Ball just changed possession (potential rebound/spill scenario)
                    elif (rew_index in self._previous_ball_owned_team and 
                          self._previous_ball_owned_team[rew_index] != ball_owned_team and
                          ball_owned_team == 0):  # Our team got the ball
                        components["opportunity_reward"][rew_index] = self._opportunity_reward_coeff
                        
                # 3. Shot attempt reward - reward shooting when in good position
                # Check if player is attempting to shoot (not dribbling/sprinting, in good position)
                sticky_actions = o.get('sticky_actions', np.zeros(10))
                is_dribbling = sticky_actions[9] == 1  # dribble action
                is_sprinting = sticky_actions[8] == 1  # sprint action
                
                # Reward being ready to shoot in danger zone (not dribbling/sprinting = ready to shoot)
                if (ball_owned_team == 0 and ball_owned_player == active_player_idx and 
                    in_danger_zone and not is_dribbling and ball_distance < 0.05):
                    components["shot_attempt_reward"][rew_index] = self._shot_attempt_coeff
                    
            # Update previous state tracking
            self._previous_ball_owned_team[rew_index] = ball_owned_team
            self._previous_ball_position[rew_index] = ball_pos.copy()
            
            # Combine all rewards
            reward[rew_index] = (components["base_score_reward"][rew_index] + 
                               components["positioning_reward"][rew_index] + 
                               components["opportunity_reward"][rew_index] + 
                               components["tap_in_bonus"][rew_index] + 
                               components["shot_attempt_reward"][rew_index])
                               
        return reward, components
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
