import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards high-speed ball carrying, breakthrough, and penalty area penetration."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._previous_positions = {}
        self._previous_ball_owned = {}
        self._total_carry_distance = {}
        self._penalty_area_entries = {}
        self._high_speed_carries = {}
        
        # Reward coefficients for easy tuning
        self._carry_distance_coeff = 0.5
        self._high_speed_coeff = 0.3
        self._penalty_entry_coeff = 1.0
        self._dribble_coeff = 0.8
        self._forward_progress_coeff = 0.4
        
        # Thresholds
        self._high_speed_threshold = 0.015  # Movement threshold for high-speed detection
        self._penalty_area_x = 0.7  # X coordinate threshold for penalty area (closer to goal at x=1)
        self._penalty_area_y = 0.2   # Y coordinate threshold for penalty area width
        
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_positions = {}
        self._previous_ball_owned = {}
        self._total_carry_distance = {}
        self._penalty_area_entries = {}
        self._high_speed_carries = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_positions': self._previous_positions,
            'previous_ball_owned': self._previous_ball_owned,
            'total_carry_distance': self._total_carry_distance,
            'penalty_area_entries': self._penalty_area_entries,
            'high_speed_carries': self._high_speed_carries
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self._previous_positions = wrapper_state['previous_positions']
        self._previous_ball_owned = wrapper_state['previous_ball_owned']
        self._total_carry_distance = wrapper_state['total_carry_distance']
        self._penalty_area_entries = wrapper_state['penalty_area_entries']
        self._high_speed_carries = wrapper_state['high_speed_carries']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_carry_reward": [0.0] * len(reward),
            "high_speed_reward": [0.0] * len(reward),
            "penalty_entry_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "forward_progress_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Initialize tracking variables for this agent if not exists
            if rew_index not in self._previous_positions:
                self._previous_positions[rew_index] = None
                self._previous_ball_owned[rew_index] = False
                self._total_carry_distance[rew_index] = 0.0
                self._penalty_area_entries[rew_index] = False
                self._high_speed_carries[rew_index] = 0
            
            # Check if the active player has the ball (left team = 0)
            player_has_ball = (
                'ball_owned_team' in o and 
                o['ball_owned_team'] == 0 and 
                'ball_owned_player' in o and 
                o['ball_owned_player'] == o['active']
            )
            
            if player_has_ball and 'left_team' in o and o['active'] < len(o['left_team']):
                current_pos = o['left_team'][o['active']]
                
                # Calculate movement and rewards only if we have a previous position
                if self._previous_positions[rew_index] is not None and self._previous_ball_owned[rew_index]:
                    prev_pos = self._previous_positions[rew_index]
                    
                    # Calculate movement distance
                    movement_distance = ((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)**0.5
                    
                    # Ball carrying distance reward
                    if movement_distance > 0:
                        components["ball_carry_reward"][rew_index] = movement_distance * self._carry_distance_coeff
                        self._total_carry_distance[rew_index] += movement_distance
                    
                    # High-speed carrying reward (sprinting with ball)
                    if movement_distance > self._high_speed_threshold:
                        components["high_speed_reward"][rew_index] = movement_distance * self._high_speed_coeff
                        self._high_speed_carries[rew_index] += 1
                    
                    # Forward progress reward (moving towards opponent goal at x=1)
                    forward_progress = current_pos[0] - prev_pos[0]
                    if forward_progress > 0:
                        components["forward_progress_reward"][rew_index] = forward_progress * self._forward_progress_coeff
                    
                    # Dribble reward - reward for maintaining ball possession while moving
                    if movement_distance > 0.005:  # Minimum movement to count as dribble
                        components["dribble_reward"][rew_index] = movement_distance * self._dribble_coeff
                
                # Penalty area entry reward
                in_penalty_area = (
                    current_pos[0] > self._penalty_area_x and 
                    abs(current_pos[1]) < self._penalty_area_y
                )
                
                if in_penalty_area and not self._penalty_area_entries[rew_index]:
                    components["penalty_entry_reward"][rew_index] = self._penalty_entry_coeff
                    self._penalty_area_entries[rew_index] = True
                
                # Update previous position
                self._previous_positions[rew_index] = current_pos.copy()
                self._previous_ball_owned[rew_index] = True
                
            else:
                # Player doesn't have ball
                if 'left_team' in o and o['active'] < len(o['left_team']):
                    self._previous_positions[rew_index] = o['left_team'][o['active']].copy()
                self._previous_ball_owned[rew_index] = False
            
            # Apply all reward components
            total_additional_reward = (
                components["ball_carry_reward"][rew_index] +
                components["high_speed_reward"][rew_index] +
                components["penalty_entry_reward"][rew_index] +
                components["dribble_reward"][rew_index] +
                components["forward_progress_reward"][rew_index]
            )
            
            reward[rew_index] += total_additional_reward

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
