import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for long-range shooting behavior."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._last_ball_owned_player = {}
        self._last_ball_position = {}
        self._shot_positions = {}
        self._penalty_arc_distance = 0.6  # Distance from goal defining penalty arc
        self._long_range_threshold = 0.4   # Minimum distance for long-range shot
        
        # Reward coefficients for easy tuning
        self._long_shot_reward = 0.3      # Reward for taking long-range shots
        self._shot_quality_reward = 0.2   # Bonus for good shot angle/position
        self._goal_distance_bonus = 0.5   # Extra bonus for long-range goals
        self._quick_shot_reward = 0.1     # Reward for quick shooting after possession
        
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._last_ball_owned_player = {}
        self._last_ball_position = {}
        self._shot_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_owned_player': self._last_ball_owned_player,
            'last_ball_position': self._last_ball_position,
            'shot_positions': self._shot_positions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self._last_ball_owned_player = wrapper_state['last_ball_owned_player']
        self._last_ball_position = wrapper_state['last_ball_position']
        self._shot_positions = wrapper_state['shot_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_shot_reward": [0.0] * len(reward),
            "shot_quality_reward": [0.0] * len(reward),
            "goal_distance_bonus": [0.0] * len(reward),
            "quick_shot_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Handle goal scoring with distance bonus
            if reward[rew_index] == 1:
                # Check if this was a long-range goal
                if rew_index in self._shot_positions:
                    shot_distance = self._shot_positions[rew_index]
                    if shot_distance >= self._long_range_threshold:
                        # Extra bonus for long-range goals
                        distance_bonus = min(shot_distance * self._goal_distance_bonus, 1.0)
                        components["goal_distance_bonus"][rew_index] = distance_bonus
                        reward[rew_index] += distance_bonus
                    # Clear shot position after goal
                    del self._shot_positions[rew_index]
                continue

            # Track ball ownership changes and detect shots
            current_ball_owned_team = o.get('ball_owned_team', -1)
            current_ball_owned_player = o.get('ball_owned_player', -1)
            current_ball_pos = o['ball']
            
            # Check if our team (left team = 0) has the ball
            if current_ball_owned_team == 0:
                ball_distance_to_goal = ((current_ball_pos[0] - 1) ** 2 + current_ball_pos[1] ** 2) ** 0.5
                
                # Store ball position for shot detection
                self._last_ball_position[rew_index] = current_ball_pos.copy()
                
                # Check if active player just gained possession (for quick shot detection)
                last_owned_player = self._last_ball_owned_player.get(rew_index, -1)
                if (last_owned_player != current_ball_owned_player and 
                    current_ball_owned_player == o.get('active', -1)):
                    # Player just gained possession - start tracking for quick shots
                    self._last_ball_owned_player[rew_index] = current_ball_owned_player
            
            # Detect potential shots (ball no longer owned by our team but was recently)
            elif (current_ball_owned_team != 0 and 
                  rew_index in self._last_ball_position and
                  rew_index in self._last_ball_owned_player):
                
                last_ball_pos = self._last_ball_position[rew_index]
                shot_distance = ((last_ball_pos[0] - 1) ** 2 + last_ball_pos[1] ** 2) ** 0.5
                
                # Check if ball is moving towards goal (indicating a shot)
                ball_direction = o.get('ball_direction', [0, 0, 0])
                moving_towards_goal = ball_direction[0] > 0  # Moving right towards opponent goal
                
                # Check if this was a long-range shot
                if (shot_distance >= self._long_range_threshold and 
                    moving_towards_goal and
                    abs(ball_direction[0]) > 0.01):  # Minimum speed threshold
                    
                    # Basic long-range shot reward
                    components["long_shot_reward"][rew_index] = self._long_shot_reward
                    reward[rew_index] += self._long_shot_reward
                    
                    # Shot quality bonus based on angle and distance
                    # Better shots are from closer to center and reasonable distance
                    angle_quality = max(0, 1 - abs(last_ball_pos[1]) * 5)  # Better if closer to center
                    distance_quality = min(1, shot_distance / 0.8)  # Normalize distance factor
                    quality_score = (angle_quality + distance_quality) / 2
                    
                    quality_bonus = quality_score * self._shot_quality_reward
                    components["shot_quality_reward"][rew_index] = quality_bonus
                    reward[rew_index] += quality_bonus
                    
                    # Quick shot bonus (if shot taken soon after gaining possession)
                    # This encourages the "lone wolf" trait of quick shooting
                    if self._last_ball_owned_player[rew_index] == o.get('active', -1):
                        components["quick_shot_reward"][rew_index] = self._quick_shot_reward
                        reward[rew_index] += self._quick_shot_reward
                    
                    # Store shot distance for potential goal bonus later
                    self._shot_positions[rew_index] = shot_distance
                
                # Clear tracking data after shot detection
                if rew_index in self._last_ball_position:
                    del self._last_ball_position[rew_index]
                if rew_index in self._last_ball_owned_player:
                    del self._last_ball_owned_player[rew_index]

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
