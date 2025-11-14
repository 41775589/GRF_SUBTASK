import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards dribbling skills, ball retention, and space creation."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._prev_ball_position = None
        self._prev_active_player_pos = None
        self._ball_possession_steps = 0
        self._dribble_sequence_length = 0
        self._last_ball_owned_player = -1
        
        # Reward coefficients for easy tuning
        self._ball_retention_coeff = 0.02
        self._dribbling_progress_coeff = 0.05
        self._penalty_area_bonus_coeff = 0.1
        self._dribble_sequence_coeff = 0.03
        self._defender_evasion_coeff = 0.08
        
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._prev_ball_position = None
        self._prev_active_player_pos = None
        self._ball_possession_steps = 0
        self._dribble_sequence_length = 0
        self._last_ball_owned_player = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'prev_ball_position': self._prev_ball_position,
            'prev_active_player_pos': self._prev_active_player_pos,
            'ball_possession_steps': self._ball_possession_steps,
            'dribble_sequence_length': self._dribble_sequence_length,
            'last_ball_owned_player': self._last_ball_owned_player
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        checkpoint_data = from_pickle['CheckpointRewardWrapper']
        self._prev_ball_position = checkpoint_data['prev_ball_position']
        self._prev_active_player_pos = checkpoint_data['prev_active_player_pos']
        self._ball_possession_steps = checkpoint_data['ball_possession_steps']
        self._dribble_sequence_length = checkpoint_data['dribble_sequence_length']
        self._last_ball_owned_player = checkpoint_data['last_ball_owned_player']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_retention_reward": [0.0] * len(reward),
            "dribbling_progress_reward": [0.0] * len(reward),
            "penalty_area_bonus": [0.0] * len(reward),
            "dribble_sequence_reward": [0.0] * len(reward),
            "defender_evasion_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reset base reward but keep original score rewards
            if reward[rew_index] != 0:
                components["base_score_reward"][rew_index] = reward[rew_index]
            else:
                components["base_score_reward"][rew_index] = 0.0

            # Check if our team (left team, team 0) has the ball
            has_ball = (o.get('ball_owned_team', -1) == 0 and 
                       o.get('ball_owned_player', -1) >= 0)
            
            current_ball_pos = o.get('ball', [0, 0, 0])
            active_player_idx = o.get('active', 0)
            left_team_pos = o.get('left_team', [])
            right_team_pos = o.get('right_team', [])
            
            if len(left_team_pos) > active_player_idx:
                current_active_pos = left_team_pos[active_player_idx]
            else:
                current_active_pos = [0, 0]

            if has_ball:
                # Ball retention reward - reward for maintaining possession
                self._ball_possession_steps += 1
                components["ball_retention_reward"][rew_index] = self._ball_retention_coeff * min(self._ball_possession_steps / 10.0, 1.0)
                
                # Check if player is dribbling (using dribble sticky action)
                is_dribbling = o.get('sticky_actions', [0]*10)[9] == 1  # dribble action
                
                if is_dribbling:
                    self._dribble_sequence_length += 1
                    # Reward for maintaining dribble sequence
                    components["dribble_sequence_reward"][rew_index] = self._dribble_sequence_coeff * min(self._dribble_sequence_length / 5.0, 1.0)
                else:
                    self._dribble_sequence_length = max(0, self._dribble_sequence_length - 1)
                
                # Dribbling progress toward goal
                if self._prev_ball_position is not None:
                    # Calculate progress toward opponent goal (right side, x=1)
                    prev_distance_to_goal = ((self._prev_ball_position[0] - 1.0) ** 2 + self._prev_ball_position[1] ** 2) ** 0.5
                    curr_distance_to_goal = ((current_ball_pos[0] - 1.0) ** 2 + current_ball_pos[1] ** 2) ** 0.5
                    
                    progress = prev_distance_to_goal - curr_distance_to_goal
                    if progress > 0:
                        components["dribbling_progress_reward"][rew_index] = self._dribbling_progress_coeff * progress
                
                # Penalty area bonus - reward for ball control in attacking third
                if current_ball_pos[0] > 0.6:  # In attacking third
                    penalty_area_multiplier = 1.0
                    if current_ball_pos[0] > 0.8:  # Close to penalty area
                        penalty_area_multiplier = 2.0
                    if current_ball_pos[0] > 0.9 and abs(current_ball_pos[1]) < 0.2:  # In penalty area
                        penalty_area_multiplier = 3.0
                    
                    components["penalty_area_bonus"][rew_index] = self._penalty_area_bonus_coeff * penalty_area_multiplier
                
                # Defender evasion reward - reward for maintaining ball possession while close to defenders
                if len(right_team_pos) > 0:
                    min_defender_distance = float('inf')
                    for defender_pos in right_team_pos:
                        distance = ((current_ball_pos[0] - defender_pos[0]) ** 2 + (current_ball_pos[1] - defender_pos[1]) ** 2) ** 0.5
                        min_defender_distance = min(min_defender_distance, distance)
                    
                    # Reward for maintaining possession under pressure (close to defenders)
                    if min_defender_distance < 0.1:  # Very close to defender
                        components["defender_evasion_reward"][rew_index] = self._defender_evasion_coeff * 2.0
                    elif min_defender_distance < 0.2:  # Moderately close
                        components["defender_evasion_reward"][rew_index] = self._defender_evasion_coeff * 1.0

            else:
                # Reset counters when ball is lost
                self._ball_possession_steps = 0
                self._dribble_sequence_length = 0

            # Update previous positions
            self._prev_ball_position = current_ball_pos.copy()
            self._prev_active_player_pos = current_active_pos.copy()
            self._last_ball_owned_player = o.get('ball_owned_player', -1)
            
            # Calculate final reward
            reward[rew_index] = (components["base_score_reward"][rew_index] + 
                               components["ball_retention_reward"][rew_index] +
                               components["dribbling_progress_reward"][rew_index] +
                               components["penalty_area_bonus"][rew_index] +
                               components["dribble_sequence_reward"][rew_index] +
                               components["defender_evasion_reward"][rew_index])

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
