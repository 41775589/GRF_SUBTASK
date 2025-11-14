import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards goal-mouth positioning and tap-in opportunities."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._dangerous_zone_time = {}  # Track time spent in dangerous zones
        self._last_ball_owned_team = {}  # Track ball possession changes
        self._last_ball_position = {}  # Track ball movement for rebound detection
        self._shots_taken = {}  # Track shots from dangerous positions
        self._goals_from_danger_zone = {}  # Track successful tap-ins
        
        # Reward coefficients - easily adjustable
        self._danger_zone_coeff = 0.01  # Reward per step in danger zone
        self._positioning_coeff = 0.05  # Reward for optimal positioning
        self._rebound_opportunity_coeff = 0.1  # Reward for being near loose balls
        self._tap_in_bonus_coeff = 0.5  # Extra reward for close-range goals
        
        # Zone definitions
        self._danger_zone_x_min = 0.7  # Close to right goal (opponent's goal)
        self._danger_zone_y_range = 0.2  # Within goal area width
        self._close_range_distance = 0.3  # Distance threshold for tap-ins
        
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._dangerous_zone_time = {}
        self._last_ball_owned_team = {}
        self._last_ball_position = {}
        self._shots_taken = {}
        self._goals_from_danger_zone = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'dangerous_zone_time': self._dangerous_zone_time,
            'last_ball_owned_team': self._last_ball_owned_team,
            'last_ball_position': self._last_ball_position,
            'shots_taken': self._shots_taken,
            'goals_from_danger_zone': self._goals_from_danger_zone
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        checkpoint_state = from_pickle['CheckpointRewardWrapper']
        self._dangerous_zone_time = checkpoint_state['dangerous_zone_time']
        self._last_ball_owned_team = checkpoint_state['last_ball_owned_team']
        self._last_ball_position = checkpoint_state['last_ball_position']
        self._shots_taken = checkpoint_state['shots_taken']
        self._goals_from_danger_zone = checkpoint_state['goals_from_danger_zone']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "danger_zone_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward),
            "rebound_opportunity_reward": [0.0] * len(reward),
            "tap_in_bonus_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Initialize tracking variables for this agent if not exists
            if rew_index not in self._dangerous_zone_time:
                self._dangerous_zone_time[rew_index] = 0
                self._last_ball_owned_team[rew_index] = o.get('ball_owned_team', -1)
                self._last_ball_position[rew_index] = o['ball'][:2].copy()
                self._shots_taken[rew_index] = 0
                self._goals_from_danger_zone[rew_index] = 0
            
            # Check for goal and if it's a tap-in (close range goal)
            if reward[rew_index] == 1:  # Goal scored
                ball_pos = o['ball'][:2]
                goal_distance = ((ball_pos[0] - 1.0) ** 2 + ball_pos[1] ** 2) ** 0.5
                
                if goal_distance <= self._close_range_distance:
                    # This is a tap-in! Give bonus reward
                    components["tap_in_bonus_reward"][rew_index] = self._tap_in_bonus_coeff
                    self._goals_from_danger_zone[rew_index] += 1
                
                # Apply all rewards for goal
                reward[rew_index] = (components["base_score_reward"][rew_index] + 
                                   components["tap_in_bonus_reward"][rew_index])
                continue

            # Get active player position
            active_player_idx = o['active']
            player_pos = o['left_team'][active_player_idx]
            ball_pos = o['ball'][:2]
            
            # 1. Danger Zone Positioning Reward
            # Check if player is in dangerous area near opponent's goal
            if (player_pos[0] >= self._danger_zone_x_min and 
                abs(player_pos[1]) <= self._danger_zone_y_range):
                components["danger_zone_reward"][rew_index] = self._danger_zone_coeff
                self._dangerous_zone_time[rew_index] += 1
            
            # 2. Optimal Positioning Reward
            # Reward for being well-positioned relative to ball and goal
            goal_pos = [1.0, 0.0]  # Right goal center
            distance_to_goal = ((player_pos[0] - goal_pos[0]) ** 2 + 
                              (player_pos[1] - goal_pos[1]) ** 2) ** 0.5
            distance_to_ball = ((player_pos[0] - ball_pos[0]) ** 2 + 
                              (player_pos[1] - ball_pos[1]) ** 2) ** 0.5
            
            # Reward for being close to goal but not too far from ball
            if distance_to_goal <= 0.4 and distance_to_ball <= 0.5:
                positioning_reward = self._positioning_coeff * (0.4 - distance_to_goal) * (0.5 - distance_to_ball)
                components["positioning_reward"][rew_index] = max(0, positioning_reward)
            
            # 3. Rebound Opportunity Reward
            # Detect potential rebounds (ball ownership changes or ball movement without clear possession)
            current_ball_owned_team = o.get('ball_owned_team', -1)
            last_ball_owned_team = self._last_ball_owned_team[rew_index]
            
            # Check if ball became loose or changed possession
            ball_is_loose = current_ball_owned_team == -1
            possession_changed = (current_ball_owned_team != last_ball_owned_team and 
                                last_ball_owned_team != -1)
            
            # Check ball movement (potential rebound)
            ball_moved_significantly = (
                ((ball_pos[0] - self._last_ball_position[rew_index][0]) ** 2 + 
                 (ball_pos[1] - self._last_ball_position[rew_index][1]) ** 2) ** 0.5 > 0.1
            )
            
            # Reward for being close to loose balls or rebounds in dangerous area
            if (ball_is_loose or possession_changed) and ball_moved_significantly:
                if (ball_pos[0] >= self._danger_zone_x_min and 
                    abs(ball_pos[1]) <= self._danger_zone_y_range * 1.5):  # Slightly larger area for rebounds
                    
                    if distance_to_ball <= 0.2:  # Very close to the loose ball
                        components["rebound_opportunity_reward"][rew_index] = self._rebound_opportunity_coeff
            
            # Update tracking variables
            self._last_ball_owned_team[rew_index] = current_ball_owned_team
            self._last_ball_position[rew_index] = ball_pos.copy()
            
            # Sum all reward components
            total_additional_reward = (components["danger_zone_reward"][rew_index] + 
                                     components["positioning_reward"][rew_index] + 
                                     components["rebound_opportunity_reward"][rew_index])
            
            reward[rew_index] = components["base_score_reward"][rew_index] + total_additional_reward

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
