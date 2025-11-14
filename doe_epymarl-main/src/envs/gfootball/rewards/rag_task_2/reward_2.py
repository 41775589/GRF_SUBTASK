import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for dribbling, ball retention, and space creation."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._previous_ball_position = None
        self._previous_player_position = None
        self._ball_possession_steps = 0
        self._dribbling_reward_coeff = 0.05
        self._retention_reward_coeff = 0.02
        self._penalty_area_reward_coeff = 0.1
        self._attacking_third_reward_coeff = 0.03
        self._space_creation_reward_coeff = 0.08
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
            "ball_retention_reward": [0.0] * len(reward),
            "penalty_area_reward": [0.0] * len(reward),
            "attacking_third_reward": [0.0] * len(reward),
            "space_creation_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # If agent scored, give full reward and reset tracking
            if reward[rew_index] == 1:
                reward[rew_index] = 1 * components["base_score_reward"][rew_index]
                self._previous_ball_position = None
                self._previous_player_position = None
                self._ball_possession_steps = 0
                continue

            # Check if our team (left team = 0) has the ball and active player controls it
            has_ball = (o.get('ball_owned_team') == 0 and 
                       o.get('ball_owned_player') == o.get('active', -1))
            
            if not has_ball:
                # Reset possession tracking if we lose the ball
                self._ball_possession_steps = 0
                self._previous_ball_position = None
                self._previous_player_position = None
                continue

            # Get current positions
            active_player_idx = o['active']
            player_pos = o['left_team'][active_player_idx]
            ball_pos = o['ball'][:2]  # x, y coordinates only
            
            # Ball retention reward - reward for maintaining possession
            self._ball_possession_steps += 1
            if self._ball_possession_steps > 3:  # Only reward after maintaining possession for a few steps
                components["ball_retention_reward"][rew_index] = self._retention_reward_coeff
                reward[rew_index] += components["ball_retention_reward"][rew_index]

            # Dribbling reward - detect if player is using dribble action and moving with ball
            is_dribbling = o['sticky_actions'][9] == 1  # dribble action is at index 9
            if is_dribbling and self._previous_player_position is not None:
                # Calculate movement distance
                player_movement = ((player_pos[0] - self._previous_player_position[0]) ** 2 + 
                                 (player_pos[1] - self._previous_player_position[1]) ** 2) ** 0.5
                
                # Reward dribbling movement, especially in tight spaces
                if player_movement > 0.005:  # Minimum movement threshold
                    components["dribbling_reward"][rew_index] = self._dribbling_reward_coeff * min(player_movement * 10, 1.0)
                    reward[rew_index] += components["dribbling_reward"][rew_index]

            # Attacking third reward - reward for ball possession in opponent's half
            if ball_pos[0] > 0:  # Right half of field (opponent's half)
                components["attacking_third_reward"][rew_index] = self._attacking_third_reward_coeff
                reward[rew_index] += components["attacking_third_reward"][rew_index]
                
                # Penalty area reward - extra reward for possession near opponent's goal
                # Penalty area is approximately from x=0.83 to x=1.0, y=-0.2 to y=0.2
                if ball_pos[0] > 0.83 and abs(ball_pos[1]) < 0.2:
                    components["penalty_area_reward"][rew_index] = self._penalty_area_reward_coeff
                    reward[rew_index] += components["penalty_area_reward"][rew_index]

            # Space creation reward - reward for moving towards goal while maintaining possession
            if self._previous_ball_position is not None:
                # Calculate progress towards opponent's goal (x = 1)
                previous_goal_distance = abs(1.0 - self._previous_ball_position[0])
                current_goal_distance = abs(1.0 - ball_pos[0])
                
                # Reward progress towards goal
                if current_goal_distance < previous_goal_distance:
                    progress = previous_goal_distance - current_goal_distance
                    components["space_creation_reward"][rew_index] = self._space_creation_reward_coeff * min(progress * 5, 0.1)
                    reward[rew_index] += components["space_creation_reward"][rew_index]
                
                # Additional reward for lateral movement in attacking third (creating angles)
                if ball_pos[0] > 0.5:  # In attacking half
                    lateral_movement = abs(ball_pos[1] - self._previous_ball_position[1])
                    if lateral_movement > 0.01:  # Significant lateral movement
                        components["space_creation_reward"][rew_index] += self._space_creation_reward_coeff * 0.3
                        reward[rew_index] += self._space_creation_reward_coeff * 0.3

            # Update tracking variables
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
