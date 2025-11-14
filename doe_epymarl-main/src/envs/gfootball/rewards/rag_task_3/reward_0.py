import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards goal-mouth positioning and tap-in opportunities."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._danger_zone_time = {}  # Track time spent in danger zone
        self._last_ball_position = {}  # Track ball movement for rebound detection
        self._shots_taken = {}  # Track shots for conversion efficiency
        self._danger_zone_threshold = 0.3  # Distance from goal to be in danger zone
        self._danger_zone_reward_coeff = 0.01  # Reward per step in danger zone
        self._tap_in_reward_coeff = 0.5  # Extra reward for close-range goals
        self._positioning_reward_coeff = 0.02  # Reward for good positioning
        self._rebound_opportunity_coeff = 0.1  # Reward for being near ball after rebounds
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._danger_zone_time = {}
        self._last_ball_position = {}
        self._shots_taken = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'danger_zone_time': self._danger_zone_time,
            'last_ball_position': self._last_ball_position,
            'shots_taken': self._shots_taken
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self._danger_zone_time = wrapper_state['danger_zone_time']
        self._last_ball_position = wrapper_state['last_ball_position']
        self._shots_taken = wrapper_state['shots_taken']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "danger_zone_reward": [0.0] * len(reward),
            "tap_in_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward),
            "rebound_opportunity_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Initialize tracking variables if needed
            if rew_index not in self._danger_zone_time:
                self._danger_zone_time[rew_index] = 0
                self._last_ball_position[rew_index] = None
                self._shots_taken[rew_index] = 0

            # Get current player position and ball position
            current_player_pos = o['left_team'][o['active']]
            ball_pos = o['ball'][:2]  # Only x, y coordinates
            
            # Calculate distance to opponent's goal (right goal at x=1)
            goal_pos = [1.0, 0.0]
            distance_to_goal = ((current_player_pos[0] - goal_pos[0])**2 + 
                              (current_player_pos[1] - goal_pos[1])**2)**0.5
            
            # Check if goal was scored for tap-in reward
            if reward[rew_index] == 1:  # Goal scored
                # Give extra reward for tap-ins (goals scored from very close range)
                if distance_to_goal < 0.2:  # Very close to goal
                    components["tap_in_reward"][rew_index] = self._tap_in_reward_coeff
                    reward[rew_index] += self._tap_in_reward_coeff
                
                # Reset counters after scoring
                self._danger_zone_time[rew_index] = 0
                continue

            # Danger zone positioning reward
            if distance_to_goal < self._danger_zone_threshold:
                self._danger_zone_time[rew_index] += 1
                components["danger_zone_reward"][rew_index] = self._danger_zone_reward_coeff
                reward[rew_index] += self._danger_zone_reward_coeff

            # Positioning reward - reward for being in good attacking positions
            # Good position: close to goal, between goal posts (-0.044 to 0.044 y range extended)
            if (current_player_pos[0] > 0.6 and  # In attacking half
                abs(current_player_pos[1]) < 0.2 and  # Near goal mouth area
                distance_to_goal < 0.4):  # Close enough to goal
                components["positioning_reward"][rew_index] = self._positioning_reward_coeff
                reward[rew_index] += self._positioning_reward_coeff

            # Rebound opportunity reward
            if self._last_ball_position[rew_index] is not None:
                # Check if ball moved significantly (possible rebound/loose ball)
                last_ball_pos = self._last_ball_position[rew_index]
                ball_movement = ((ball_pos[0] - last_ball_pos[0])**2 + 
                               (ball_pos[1] - last_ball_pos[1])**2)**0.5
                
                # If ball moved significantly and player is close to it in danger zone
                if (ball_movement > 0.05 and  # Ball moved significantly
                    distance_to_goal < self._danger_zone_threshold and  # In danger zone
                    ((current_player_pos[0] - ball_pos[0])**2 + 
                     (current_player_pos[1] - ball_pos[1])**2)**0.5 < 0.1):  # Close to ball
                    components["rebound_opportunity_reward"][rew_index] = self._rebound_opportunity_coeff
                    reward[rew_index] += self._rebound_opportunity_coeff

            # Update ball position tracking
            self._last_ball_position[rew_index] = ball_pos.copy()

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
