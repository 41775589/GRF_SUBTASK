import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards goal-mouth positioning and tap-in opportunities."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._dangerous_zone_time = {}  # Track time spent in dangerous zones
        self._last_ball_position = {}   # Track ball movement for rebound detection
        self._last_shot_distance = {}   # Track distance of last shot for conversion efficiency
        self._dangerous_zone_threshold = 0.3  # Distance from goal to be considered dangerous
        self._positioning_reward = 0.01  # Reward per step in dangerous zone
        self._tap_in_bonus = 0.5  # Bonus for scoring from very close range
        self._conversion_bonus = 0.3  # Bonus for efficient shot conversion
        self._rebound_bonus = 0.2  # Bonus for capitalizing on rebounds
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._dangerous_zone_time = {}
        self._last_ball_position = {}
        self._last_shot_distance = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'dangerous_zone_time': self._dangerous_zone_time,
            'last_ball_position': self._last_ball_position,
            'last_shot_distance': self._last_shot_distance
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self._dangerous_zone_time = wrapper_state['dangerous_zone_time']
        self._last_ball_position = wrapper_state['last_ball_position']
        self._last_shot_distance = wrapper_state['last_shot_distance']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0] * len(reward),
            "tap_in_reward": [0.0] * len(reward),
            "conversion_reward": [0.0] * len(reward),
            "rebound_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Initialize tracking variables if not present
            if rew_index not in self._dangerous_zone_time:
                self._dangerous_zone_time[rew_index] = 0
            if rew_index not in self._last_ball_position:
                self._last_ball_position[rew_index] = o['ball'][:2].copy()
            if rew_index not in self._last_shot_distance:
                self._last_shot_distance[rew_index] = None

            # Check if agent scored a goal
            if reward[rew_index] == 1:
                # Calculate distance from goal for tap-in bonus
                active_player_pos = o['left_team'][o['active']]
                goal_distance = abs(1.0 - active_player_pos[0])  # Distance to right goal
                
                # Tap-in bonus for very close goals
                if goal_distance <= 0.15:  # Very close to goal
                    components["tap_in_reward"][rew_index] = self._tap_in_bonus
                    reward[rew_index] += components["tap_in_reward"][rew_index]
                
                # Conversion efficiency bonus based on shot distance
                if self._last_shot_distance[rew_index] is not None:
                    if self._last_shot_distance[rew_index] <= 0.25:  # Close range shot
                        components["conversion_reward"][rew_index] = self._conversion_bonus
                        reward[rew_index] += components["conversion_reward"][rew_index]
                
                # Check for rebound opportunity (ball movement indicates goalkeeper action)
                ball_movement = np.linalg.norm(o['ball'][:2] - self._last_ball_position[rew_index])
                if ball_movement > 0.1 and goal_distance <= 0.2:  # Ball moved significantly and player is close
                    components["rebound_reward"][rew_index] = self._rebound_bonus
                    reward[rew_index] += components["rebound_reward"][rew_index]
                
                continue

            # Track positioning in dangerous zones
            if o['active'] < len(o['left_team']):
                active_player_pos = o['left_team'][o['active']]
                
                # Calculate distance from opponent's goal (right goal at x=1)
                goal_distance = ((1.0 - active_player_pos[0]) ** 2 + (active_player_pos[1]) ** 2) ** 0.5
                
                # Reward for positioning in dangerous zone near goal
                if goal_distance <= self._dangerous_zone_threshold:
                    # Extra reward for being in the penalty area (closer to goal)
                    if goal_distance <= 0.15 and abs(active_player_pos[1]) <= 0.1:  # Very close to goal mouth
                        components["positioning_reward"][rew_index] = self._positioning_reward * 2.0
                    else:
                        components["positioning_reward"][rew_index] = self._positioning_reward
                    
                    reward[rew_index] += components["positioning_reward"][rew_index]
                    self._dangerous_zone_time[rew_index] += 1
                
                # Track shot attempts for conversion efficiency
                # Detect potential shot by checking if player has ball and is in shooting position
                if (o['ball_owned_team'] == 0 and 
                    o['ball_owned_player'] == o['active'] and
                    goal_distance <= 0.4):  # Player has ball in shooting range
                    self._last_shot_distance[rew_index] = goal_distance
            
            # Update ball position tracking for rebound detection
            self._last_ball_position[rew_index] = o['ball'][:2].copy()

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
