import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards high-speed ball carrying, breakthrough dribbles, and penalty area shots."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._ball_carry_distance = {}
        self._previous_ball_position = {}
        self._previous_player_position = {}
        self._dribble_count = {}
        self._pass_penalty = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Reward coefficients for easy adjustment
        self._distance_reward_coeff = 0.5
        self._sprint_bonus_coeff = 0.3
        self._dribble_reward_coeff = 0.4
        self._penalty_shot_reward_coeff = 1.0
        self._pass_penalty_coeff = -0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_carry_distance = {}
        self._previous_ball_position = {}
        self._previous_player_position = {}
        self._dribble_count = {}
        self._pass_penalty = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'ball_carry_distance': self._ball_carry_distance,
            'previous_ball_position': self._previous_ball_position,
            'previous_player_position': self._previous_player_position,
            'dribble_count': self._dribble_count,
            'pass_penalty': self._pass_penalty
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self._ball_carry_distance = wrapper_state['ball_carry_distance']
        self._previous_ball_position = wrapper_state['previous_ball_position']
        self._previous_player_position = wrapper_state['previous_player_position']
        self._dribble_count = wrapper_state['dribble_count']
        self._pass_penalty = wrapper_state['pass_penalty']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_carry_reward": [0.0] * len(reward),
            "sprint_bonus_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "penalty_shot_reward": [0.0] * len(reward),
            "pass_penalty_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Initialize tracking variables for this agent if not exists
            if rew_index not in self._ball_carry_distance:
                self._ball_carry_distance[rew_index] = 0.0
                self._previous_ball_position[rew_index] = None
                self._previous_player_position[rew_index] = None
                self._dribble_count[rew_index] = 0
                self._pass_penalty[rew_index] = 0.0

            # Check if goal was scored - give full reward
            if reward[rew_index] == 1:
                # Add accumulated carry distance and dribble rewards when scoring
                components["ball_carry_reward"][rew_index] = self._ball_carry_distance[rew_index] * self._distance_reward_coeff
                components["dribble_reward"][rew_index] = self._dribble_count[rew_index] * self._dribble_reward_coeff
                components["pass_penalty_reward"][rew_index] = self._pass_penalty[rew_index] * self._pass_penalty_coeff
                
                reward[rew_index] = (components["base_score_reward"][rew_index] + 
                                   components["ball_carry_reward"][rew_index] + 
                                   components["dribble_reward"][rew_index] +
                                   components["pass_penalty_reward"][rew_index])
                
                # Reset tracking for this agent
                self._ball_carry_distance[rew_index] = 0.0
                self._dribble_count[rew_index] = 0
                self._pass_penalty[rew_index] = 0.0
                continue

            # Check if the active player has the ball
            if ('ball_owned_team' not in o or o['ball_owned_team'] != 0 or
                'ball_owned_player' not in o or o['ball_owned_player'] != o['active']):
                
                # Reset tracking when player doesn't have ball
                self._previous_ball_position[rew_index] = None
                self._previous_player_position[rew_index] = None
                continue

            current_ball_pos = o['ball'][:2]  # x, y coordinates
            current_player_pos = o['left_team'][o['active']]
            
            # Ball carrying distance reward (only when moving towards goal)
            if self._previous_ball_position[rew_index] is not None:
                # Calculate distance moved towards opponent goal (x=1)
                prev_x = self._previous_ball_position[rew_index][0]
                curr_x = current_ball_pos[0]
                forward_distance = max(0, curr_x - prev_x)  # Only reward forward movement
                
                if forward_distance > 0:
                    self._ball_carry_distance[rew_index] += forward_distance
                    components["ball_carry_reward"][rew_index] = forward_distance * self._distance_reward_coeff
                    
                    # Sprint bonus - check if sprint action is active (index 8 in sticky_actions)
                    if o['sticky_actions'][8] == 1:
                        components["sprint_bonus_reward"][rew_index] = forward_distance * self._sprint_bonus_coeff
                    
                    reward[rew_index] += (components["ball_carry_reward"][rew_index] + 
                                        components["sprint_bonus_reward"][rew_index])

            # Dribble past defenders reward
            if self._previous_player_position[rew_index] is not None:
                # Check if player moved significantly while maintaining ball control
                player_movement = ((current_player_pos[0] - self._previous_player_position[rew_index][0])**2 + 
                                 (current_player_pos[1] - self._previous_player_position[rew_index][1])**2)**0.5
                
                # Check if there are nearby opponents that were potentially dribbled past
                if player_movement > 0.02:  # Significant movement threshold
                    nearby_opponents = 0
                    for opponent_pos in o['right_team']:
                        distance_to_opponent = ((current_player_pos[0] - opponent_pos[0])**2 + 
                                              (current_player_pos[1] - opponent_pos[1])**2)**0.5
                        if distance_to_opponent < 0.05:  # Close proximity threshold
                            nearby_opponents += 1
                    
                    if nearby_opponents > 0:
                        dribble_reward = nearby_opponents * self._dribble_reward_coeff
                        components["dribble_reward"][rew_index] = dribble_reward
                        self._dribble_count[rew_index] += nearby_opponents
                        reward[rew_index] += dribble_reward

            # Penalty area shot reward
            # Penalty area is roughly x > 0.8 and |y| < 0.2 for opponent goal
            if (current_ball_pos[0] > 0.8 and abs(current_ball_pos[1]) < 0.2 and
                o['game_mode'] == 0):  # Normal game mode
                
                # Check if player is likely to shoot (high forward velocity or in shooting position)
                ball_direction = o['ball_direction'][:2]
                if ball_direction[0] > 0:  # Ball moving towards goal
                    components["penalty_shot_reward"][rew_index] = self._penalty_shot_reward_coeff
                    reward[rew_index] += components["penalty_shot_reward"][rew_index]

            # Pass penalty (lone wolf behavior) - penalize when ball changes ownership within team
            # This is approximated by checking if designated player changes frequently
            if 'designated' in o and o['designated'] != o['active']:
                # Potential pass occurred, apply small penalty
                pass_penalty = self._pass_penalty_coeff * 0.1
                components["pass_penalty_reward"][rew_index] = pass_penalty
                self._pass_penalty[rew_index] += 0.1
                reward[rew_index] += pass_penalty

            # Update previous positions
            self._previous_ball_position[rew_index] = current_ball_pos.copy()
            self._previous_player_position[rew_index] = current_player_pos.copy()

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
