import gym
import numpy as np


class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for long-range shooting behavior."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._long_range_threshold = 0.3  # Distance from goal for long-range shots
        self._medium_range_threshold = 0.5  # Distance from goal for medium-range shots
        self._shot_reward = 0.3  # Base reward for taking a long-range shot
        self._quality_bonus = 0.2  # Bonus for high-quality shots
        self._goal_multiplier = 2.0  # Multiplier for scoring goals
        self._previous_ball_position = None
        self._previous_ball_owned = False
        self._shots_taken = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = None
        self._previous_ball_owned = False
        self._shots_taken = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'shots_taken': self._shots_taken,
            'previous_ball_position': self._previous_ball_position,
            'previous_ball_owned': self._previous_ball_owned
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        checkpoint_state = from_pickle['CheckpointRewardWrapper']
        self._shots_taken = checkpoint_state['shots_taken']
        self._previous_ball_position = checkpoint_state['previous_ball_position']
        self._previous_ball_owned = checkpoint_state['previous_ball_owned']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_range_shot_reward": [0.0] * len(reward),
            "shot_quality_bonus": [0.0] * len(reward),
            "goal_scoring_bonus": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if agent scored a goal - give extra bonus for long-range goals
            if reward[rew_index] == 1:
                components["goal_scoring_bonus"][rew_index] = self._goal_multiplier
                reward[rew_index] += components["goal_scoring_bonus"][rew_index]
                continue

            # Check if our team (left team, team 0) has the ball
            if ('ball_owned_team' not in o or o['ball_owned_team'] != 0 or
                    'ball_owned_player' not in o or o['ball_owned_player'] != o['active']):
                self._previous_ball_owned = False
                continue

            current_ball_pos = o['ball'][:2]  # [x, y] position
            active_player_pos = o['left_team'][o['active']]

            # Calculate distance to opponent goal (right goal at x=1)
            goal_pos = [1.0, 0.0]
            distance_to_goal = ((current_ball_pos[0] - goal_pos[0]) ** 2 +
                                (current_ball_pos[1] - goal_pos[1]) ** 2) ** 0.5

            # Detect if a shot was taken by checking ball movement towards goal
            shot_detected = False
            if (self._previous_ball_position is not None and
                    self._previous_ball_owned and
                    'ball_direction' in o):

                ball_direction = o['ball_direction'][:2]
                # Check if ball is moving significantly towards goal
                direction_to_goal = [goal_pos[0] - current_ball_pos[0],
                                     goal_pos[1] - current_ball_pos[1]]
                direction_magnitude = (direction_to_goal[0] ** 2 + direction_to_goal[1] ** 2) ** 0.5

                if direction_magnitude > 0:
                    direction_to_goal = [d / direction_magnitude for d in direction_to_goal]
                    ball_speed = (ball_direction[0] ** 2 + ball_direction[1] ** 2) ** 0.5

                    # Dot product to check if ball direction aligns with goal direction
                    alignment = (ball_direction[0] * direction_to_goal[0] +
                                 ball_direction[1] * direction_to_goal[1])

                    # Shot detected if ball is moving fast towards goal
                    if ball_speed > 0.05 and alignment > 0.7:
                        shot_detected = True

            # Reward long-range and medium-range shots
            if shot_detected:
                shot_reward = 0.0

                # Long-range shot (beyond penalty arc)
                if distance_to_goal >= self._long_range_threshold:
                    shot_reward = self._shot_reward
                    components["long_range_shot_reward"][rew_index] = shot_reward

                    # Calculate shot quality bonus based on angle and distance
                    # Better angle (closer to center) and optimal distance get higher bonus
                    angle_quality = 1.0 - abs(current_ball_pos[1]) / 0.42  # Normalize by field width
                    distance_quality = min(1.0, distance_to_goal / self._long_range_threshold)

                    quality_bonus = self._quality_bonus * angle_quality * distance_quality
                    components["shot_quality_bonus"][rew_index] = quality_bonus

                    reward[rew_index] += shot_reward + quality_bonus
                    self._shots_taken += 1

                # Medium-range shot (smaller reward)
                elif distance_to_goal >= self._medium_range_threshold:
                    shot_reward = self._shot_reward * 0.5
                    components["long_range_shot_reward"][rew_index] = shot_reward

                    angle_quality = 1.0 - abs(current_ball_pos[1]) / 0.42
                    quality_bonus = self._quality_bonus * 0.5 * angle_quality
                    components["shot_quality_bonus"][rew_index] = quality_bonus

                    reward[rew_index] += shot_reward + quality_bonus
                    self._shots_taken += 1

            # Update tracking variables
            self._previous_ball_position = current_ball_pos.copy()
            self._previous_ball_owned = True

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

