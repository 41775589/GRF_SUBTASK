import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for long-range shooting behavior."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._previous_ball_position = None
        self._previous_ball_owned = False
        self._possession_start_position = None
        self._shot_reward = 0.3
        self._goal_distance_bonus = 0.5
        self._angle_quality_reward = 0.2
        self._quick_shot_reward = 0.1
        self._possession_steps = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = None
        self._previous_ball_owned = False
        self._possession_start_position = None
        self._possession_steps = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self._previous_ball_position,
            'previous_ball_owned': self._previous_ball_owned,
            'possession_start_position': self._possession_start_position,
            'possession_steps': self._possession_steps
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        checkpoint_state = from_pickle['CheckpointRewardWrapper']
        self._previous_ball_position = checkpoint_state['previous_ball_position']
        self._previous_ball_owned = checkpoint_state['previous_ball_owned']
        self._possession_start_position = checkpoint_state['possession_start_position']
        self._possession_steps = checkpoint_state['possession_steps']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_range_shot_reward": [0.0] * len(reward),
            "goal_distance_bonus": [0.0] * len(reward),
            "shot_angle_quality_reward": [0.0] * len(reward),
            "quick_shot_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if we scored a goal
            if reward[rew_index] == 1:
                # If we scored from long range, give distance bonus
                if self._previous_ball_position is not None:
                    goal_distance = abs(self._previous_ball_position[0] - 1.0)  # Distance from right goal
                    if goal_distance > 0.3:  # Long range goal (beyond penalty area)
                        distance_bonus = min(goal_distance * self._goal_distance_bonus, 1.0)
                        components["goal_distance_bonus"][rew_index] = distance_bonus
                        reward[rew_index] += distance_bonus
                continue

            # Check if our team has the ball
            current_ball_owned = (o.get('ball_owned_team') == 0 and 
                                o.get('ball_owned_player') == o.get('active'))
            
            # Track possession start
            if current_ball_owned and not self._previous_ball_owned:
                self._possession_start_position = o['ball'][:2].copy()
                self._possession_steps = 0
            elif current_ball_owned:
                self._possession_steps += 1

            # Detect potential shots (ball moving fast towards goal)
            if ('ball_direction' in o and self._previous_ball_position is not None):
                ball_speed = np.linalg.norm(o['ball_direction'][:2])
                ball_pos = o['ball'][:2]
                
                # Check if ball is moving towards opponent goal (right side)
                if (ball_speed > 0.05 and o['ball_direction'][0] > 0.02 and 
                    ball_pos[0] < 0.7):  # Shot detection
                    
                    # Calculate shot distance from goal
                    goal_position = np.array([1.0, 0.0])  # Right goal center
                    shot_distance = np.linalg.norm(ball_pos - goal_position)
                    
                    # Reward long-range shots (beyond penalty area ~0.3 distance)
                    if shot_distance > 0.3:
                        # Distance-based shot reward
                        distance_reward = min(shot_distance * self._shot_reward, 0.8)
                        components["long_range_shot_reward"][rew_index] = distance_reward
                        reward[rew_index] += distance_reward
                        
                        # Shot angle quality reward
                        # Better angle = ball direction more aligned with goal direction
                        to_goal = goal_position - ball_pos
                        to_goal_norm = to_goal / np.linalg.norm(to_goal)
                        ball_dir_norm = o['ball_direction'][:2] / np.linalg.norm(o['ball_direction'][:2])
                        
                        # Dot product gives alignment (-1 to 1, we want close to 1)
                        angle_quality = np.dot(ball_dir_norm, to_goal_norm)
                        if angle_quality > 0.5:  # Reasonable shot angle
                            angle_reward = (angle_quality - 0.5) * self._angle_quality_reward
                            components["shot_angle_quality_reward"][rew_index] = angle_reward
                            reward[rew_index] += angle_reward
                        
                        # Quick shot reward (shooting soon after gaining possession)
                        if self._possession_steps > 0 and self._possession_steps <= 5:
                            quick_reward = (6 - self._possession_steps) * self._quick_shot_reward / 5
                            components["quick_shot_reward"][rew_index] = quick_reward
                            reward[rew_index] += quick_reward

            # Update tracking variables
            self._previous_ball_position = o['ball'][:2].copy() if 'ball' in o else None
            self._previous_ball_owned = current_ball_owned

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
