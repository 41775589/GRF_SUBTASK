import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards fine dribbling, ball retention, and space creation in attacking areas."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._previous_ball_position = None
        self._previous_player_position = None
        self._ball_possession_steps = 0
        self._max_possession_reward = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward coefficients for easy tuning
        self._dribbling_coefficient = 0.05
        self._possession_coefficient = 0.02
        self._attacking_area_coefficient = 0.08
        self._penalty_area_coefficient = 0.15
        self._space_creation_coefficient = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = None
        self._previous_player_position = None
        self._ball_possession_steps = 0
        self._max_possession_reward = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self._previous_ball_position,
            'previous_player_position': self._previous_player_position,
            'ball_possession_steps': self._ball_possession_steps,
            'max_possession_reward': self._max_possession_reward
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self._previous_ball_position = wrapper_state['previous_ball_position']
        self._previous_player_position = wrapper_state['previous_player_position']
        self._ball_possession_steps = wrapper_state['ball_possession_steps']
        self._max_possession_reward = wrapper_state['max_possession_reward']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0] * len(reward),
            "possession_reward": [0.0] * len(reward),
            "attacking_area_reward": [0.0] * len(reward),
            "penalty_area_reward": [0.0] * len(reward),
            "space_creation_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Get current positions
            ball_pos = o['ball'][:2]  # [x, y]
            active_player_idx = o['active']
            player_pos = o['left_team'][active_player_idx]
            
            # Check if our team (left team) has possession
            has_possession = (o['ball_owned_team'] == 0 and 
                            o['ball_owned_player'] == active_player_idx)
            
            if has_possession:
                self._ball_possession_steps += 1
                
                # 1. Dribbling reward - reward for controlled ball movement
                if self._previous_ball_position is not None and self._previous_player_position is not None:
                    ball_movement = ((ball_pos[0] - self._previous_ball_position[0])**2 + 
                                   (ball_pos[1] - self._previous_ball_position[1])**2)**0.5
                    player_movement = ((player_pos[0] - self._previous_player_position[0])**2 + 
                                     (player_pos[1] - self._previous_player_position[1])**2)**0.5
                    
                    # Reward controlled dribbling (ball and player moving together)
                    if ball_movement > 0.001 and player_movement > 0.001:
                        # Check if ball is moving toward goal (positive x direction)
                        if ball_pos[0] > self._previous_ball_position[0]:
                            components["dribbling_reward"][rew_index] = self._dribbling_coefficient
                            reward[rew_index] += components["dribbling_reward"][rew_index]
                
                # 2. Possession reward - reward for maintaining possession
                possession_bonus = min(self._ball_possession_steps * self._possession_coefficient, 0.2)
                if possession_bonus > self._max_possession_reward:
                    components["possession_reward"][rew_index] = possession_bonus - self._max_possession_reward
                    self._max_possession_reward = possession_bonus
                    reward[rew_index] += components["possession_reward"][rew_index]
                
                # 3. Attacking area reward - bonus for having possession in attacking third
                if ball_pos[0] > 0.33:  # Attacking third
                    components["attacking_area_reward"][rew_index] = self._attacking_area_coefficient
                    reward[rew_index] += components["attacking_area_reward"][rew_index]
                
                # 4. Penalty area reward - high bonus for possession in penalty area
                if ball_pos[0] > 0.83 and abs(ball_pos[1]) < 0.2:  # Penalty area approximation
                    components["penalty_area_reward"][rew_index] = self._penalty_area_coefficient
                    reward[rew_index] += components["penalty_area_reward"][rew_index]
                
                # 5. Space creation reward - reward for creating distance from defenders
                if len(o['right_team']) > 0:
                    min_distance_to_defender = float('inf')
                    for defender_pos in o['right_team']:
                        dist = ((player_pos[0] - defender_pos[0])**2 + 
                               (player_pos[1] - defender_pos[1])**2)**0.5
                        min_distance_to_defender = min(min_distance_to_defender, dist)
                    
                    # Reward for maintaining good distance from nearest defender
                    if min_distance_to_defender > 0.05:  # Good spacing
                        space_reward = min(min_distance_to_defender * self._space_creation_coefficient, 0.1)
                        components["space_creation_reward"][rew_index] = space_reward
                        reward[rew_index] += components["space_creation_reward"][rew_index]
                
            else:
                # Reset possession counter when ball is lost
                self._ball_possession_steps = 0
                self._max_possession_reward = 0
            
            # Update previous positions
            self._previous_ball_position = ball_pos.copy()
            self._previous_player_position = player_pos.copy()
        
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
