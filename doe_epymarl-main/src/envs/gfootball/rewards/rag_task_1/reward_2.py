import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for long-range shooting behavior."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._possession_reward = 0.05
        self._position_reward = 0.1
        self._shot_attempt_reward = 0.3
        self._shot_quality_reward = 0.2
        self._long_range_goal_bonus = 0.5
        self._penalty_arc_distance = 0.16  # Distance from goal defining penalty arc
        self._medium_range_distance = 0.3  # Medium range shooting distance
        self._previous_ball_position = None
        self._previous_possession_team = -1
        self._shots_taken = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = None
        self._previous_possession_team = -1
        self._shots_taken = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self._previous_ball_position,
            'previous_possession_team': self._previous_possession_team,
            'shots_taken': self._shots_taken
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        checkpoint_state = from_pickle['CheckpointRewardWrapper']
        self._previous_ball_position = checkpoint_state['previous_ball_position']
        self._previous_possession_team = checkpoint_state['previous_possession_team']
        self._shots_taken = checkpoint_state['shots_taken']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "possession_reward": [0.0] * len(reward),
            "position_reward": [0.0] * len(reward),
            "shot_attempt_reward": [0.0] * len(reward),
            "shot_quality_reward": [0.0] * len(reward),
            "long_range_goal_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if agent scored a goal
            if reward[rew_index] == 1:
                # Check if it was a long-range goal (ball was far from goal when last possessed)
                if self._previous_ball_position is not None:
                    prev_distance_to_goal = ((self._previous_ball_position[0] - 1) ** 2 + 
                                           self._previous_ball_position[1] ** 2) ** 0.5
                    if prev_distance_to_goal >= self._penalty_arc_distance:
                        components["long_range_goal_reward"][rew_index] = self._long_range_goal_bonus
                        reward[rew_index] += components["long_range_goal_reward"][rew_index]
                continue

            # Ball possession reward - encourage getting and keeping the ball
            if (o.get('ball_owned_team') == 0 and 
                o.get('ball_owned_player') == o.get('active')):
                components["possession_reward"][rew_index] = self._possession_reward
                
                # Calculate distance to opponent goal
                ball_pos = o['ball']
                distance_to_goal = ((ball_pos[0] - 1) ** 2 + ball_pos[1] ** 2) ** 0.5
                
                # Position reward - encourage moving to good shooting positions
                # Reward decreases as we get closer to goal (encouraging long-range positions)
                if distance_to_goal >= self._penalty_arc_distance:
                    if distance_to_goal <= self._medium_range_distance:
                        # Sweet spot for medium-range shooting
                        position_multiplier = 2.0
                    else:
                        # Still reward long-range positions but less
                        position_multiplier = 1.5
                    
                    components["position_reward"][rew_index] = self._position_reward * position_multiplier
                    
                    # Calculate shooting angle quality
                    # Better angle = more direct path to goal center
                    goal_center = [1.0, 0.0]
                    angle_to_goal = abs(ball_pos[1] - goal_center[1])
                    # Reward better angles (smaller Y deviation from goal center)
                    angle_quality = max(0, 1.0 - angle_to_goal * 5)  # Scale factor for angle importance
                    
                    if angle_quality > 0.5:  # Only reward reasonably good angles
                        components["shot_quality_reward"][rew_index] = (
                            self._shot_quality_reward * angle_quality)

            # Detect shot attempts by checking ball movement toward goal
            if self._previous_ball_position is not None and o.get('ball_owned_team') != 0:
                # Ball is no longer possessed by our team - might be a shot
                current_ball_pos = o['ball']
                prev_distance = ((self._previous_ball_position[0] - 1) ** 2 + 
                               self._previous_ball_position[1] ** 2) ** 0.5
                
                # Check if ball was in long-range shooting position and is moving toward goal
                if (prev_distance >= self._penalty_arc_distance and 
                    self._previous_possession_team == 0):
                    
                    ball_direction = o.get('ball_direction', [0, 0, 0])
                    # Check if ball is moving toward goal (positive X direction with reasonable speed)
                    if ball_direction[0] > 0.02:  # Threshold for shot detection
                        components["shot_attempt_reward"][rew_index] = self._shot_attempt_reward
                        self._shots_taken += 1
                        
                        # Bonus for shot quality based on previous position
                        angle_quality = max(0, 1.0 - abs(self._previous_ball_position[1]) * 3)
                        if angle_quality > 0.3:
                            components["shot_quality_reward"][rew_index] += (
                                self._shot_quality_reward * angle_quality * 0.5)

            # Update tracking variables
            if o.get('ball_owned_team') == 0:
                self._previous_ball_position = o['ball'].copy()
            self._previous_possession_team = o.get('ball_owned_team', -1)
            
            # Sum up all reward components
            total_component_reward = (
                components["possession_reward"][rew_index] +
                components["position_reward"][rew_index] +
                components["shot_attempt_reward"][rew_index] +
                components["shot_quality_reward"][rew_index] +
                components["long_range_goal_reward"][rew_index]
            )
            
            reward[rew_index] += total_component_reward

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
