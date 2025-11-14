import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for high-speed ball carrying, breakthrough dribbling, and penalty area penetration."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward coefficients for easy adjustment
        self._ball_carry_reward_coef = 0.05
        self._speed_bonus_coef = 0.03
        self._breakthrough_reward_coef = 0.15
        self._penalty_area_reward_coef = 0.2
        self._forward_progress_coef = 0.08
        
        # State tracking for each agent
        self._previous_ball_position = {}
        self._previous_player_position = {}
        self._previous_opponent_distances = {}
        self._penalty_area_entries = {}
        self._total_carry_distance = {}
        
        # Thresholds
        self._speed_threshold = 0.015  # Minimum speed for speed bonus
        self._breakthrough_distance_threshold = 0.08  # Distance change indicating breakthrough
        self._penalty_area_x = 0.83  # X coordinate defining penalty area start
        self._penalty_area_y = 0.22  # Y coordinate defining penalty area bounds

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = {}
        self._previous_player_position = {}
        self._previous_opponent_distances = {}
        self._penalty_area_entries = {}
        self._total_carry_distance = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self._previous_ball_position,
            'previous_player_position': self._previous_player_position,
            'previous_opponent_distances': self._previous_opponent_distances,
            'penalty_area_entries': self._penalty_area_entries,
            'total_carry_distance': self._total_carry_distance
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        checkpoint_state = from_pickle['CheckpointRewardWrapper']
        self._previous_ball_position = checkpoint_state['previous_ball_position']
        self._previous_player_position = checkpoint_state['previous_player_position']
        self._previous_opponent_distances = checkpoint_state['previous_opponent_distances']
        self._penalty_area_entries = checkpoint_state['penalty_area_entries']
        self._total_carry_distance = checkpoint_state['total_carry_distance']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_carry_reward": [0.0] * len(reward),
            "speed_bonus_reward": [0.0] * len(reward),
            "breakthrough_reward": [0.0] * len(reward),
            "penalty_area_reward": [0.0] * len(reward),
            "forward_progress_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Initialize tracking for new agents
            if rew_index not in self._previous_ball_position:
                self._previous_ball_position[rew_index] = None
                self._previous_player_position[rew_index] = None
                self._previous_opponent_distances[rew_index] = None
                self._penalty_area_entries[rew_index] = False
                self._total_carry_distance[rew_index] = 0.0

            # Check if the active player has the ball (left team = 0)
            if ('ball_owned_team' not in o or
                    o['ball_owned_team'] != 0 or
                    'ball_owned_player' not in o or
                    o['ball_owned_player'] != o['active']):
                # Reset tracking when not possessing ball
                self._previous_ball_position[rew_index] = None
                self._previous_player_position[rew_index] = None
                self._previous_opponent_distances[rew_index] = None
                continue

            current_ball_pos = o['ball'][:2]  # x, y coordinates
            current_player_pos = o['left_team'][o['active']]
            current_player_direction = o['left_team_direction'][o['active']]
            
            # Calculate current speed
            current_speed = (current_player_direction[0]**2 + current_player_direction[1]**2)**0.5
            
            # Calculate distances to all right team players (opponents)
            current_opponent_distances = []
            for opponent_pos in o['right_team']:
                dist = ((current_player_pos[0] - opponent_pos[0])**2 + 
                       (current_player_pos[1] - opponent_pos[1])**2)**0.5
                current_opponent_distances.append(dist)

            if self._previous_ball_position[rew_index] is not None:
                # Calculate ball carrying distance
                carry_distance = ((current_ball_pos[0] - self._previous_ball_position[rew_index][0])**2 + 
                                (current_ball_pos[1] - self._previous_ball_position[rew_index][1])**2)**0.5
                
                # Ball carrying reward - reward for moving with the ball
                if carry_distance > 0.005:  # Minimum movement threshold
                    components["ball_carry_reward"][rew_index] = self._ball_carry_reward_coef * carry_distance
                    self._total_carry_distance[rew_index] += carry_distance
                
                # Speed bonus - reward for high-speed carrying
                if current_speed > self._speed_threshold:
                    speed_bonus = self._speed_bonus_coef * (current_speed - self._speed_threshold)
                    components["speed_bonus_reward"][rew_index] = speed_bonus
                
                # Forward progress reward - encourage moving toward opponent goal (x = 1)
                forward_progress = current_ball_pos[0] - self._previous_ball_position[rew_index][0]
                if forward_progress > 0:
                    components["forward_progress_reward"][rew_index] = self._forward_progress_coef * forward_progress
                
                # Breakthrough reward - reward for getting past defenders
                if self._previous_opponent_distances[rew_index] is not None:
                    for i, (prev_dist, curr_dist) in enumerate(zip(self._previous_opponent_distances[rew_index], 
                                                                   current_opponent_distances)):
                        # If we were close to an opponent and now moved significantly away while carrying ball
                        if (prev_dist < 0.1 and curr_dist > prev_dist + self._breakthrough_distance_threshold and 
                            carry_distance > 0.01):
                            components["breakthrough_reward"][rew_index] += self._breakthrough_reward_coef
            
            # Penalty area penetration reward
            if (current_ball_pos[0] > self._penalty_area_x and 
                abs(current_ball_pos[1]) < self._penalty_area_y and 
                not self._penalty_area_entries[rew_index]):
                components["penalty_area_reward"][rew_index] = self._penalty_area_reward_coef
                self._penalty_area_entries[rew_index] = True
            
            # Reset penalty area entry if player leaves penalty area
            if (current_ball_pos[0] < self._penalty_area_x or 
                abs(current_ball_pos[1]) > self._penalty_area_y):
                self._penalty_area_entries[rew_index] = False
            
            # Update previous state
            self._previous_ball_position[rew_index] = current_ball_pos.copy()
            self._previous_player_position[rew_index] = current_player_pos.copy()
            self._previous_opponent_distances[rew_index] = current_opponent_distances.copy()
            
            # Combine all reward components
            total_additional_reward = (components["ball_carry_reward"][rew_index] +
                                     components["speed_bonus_reward"][rew_index] +
                                     components["breakthrough_reward"][rew_index] +
                                     components["penalty_area_reward"][rew_index] +
                                     components["forward_progress_reward"][rew_index])
            
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
