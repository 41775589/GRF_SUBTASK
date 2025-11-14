import gym
import numpy as np


class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards fine dribbling, ball retention, and space creation in attacking areas."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._previous_ball_position = None
        self._previous_player_position = None
        self._consecutive_possession_steps = 0
        self._dribbling_sequence_reward = 0.05  # Coefficient for continuous dribbling
        self._ball_retention_reward = 0.02  # Coefficient for maintaining possession
        self._attacking_area_reward = 0.03  # Coefficient for being in attacking third
        self._penalty_area_reward = 0.08  # Coefficient for penalty area presence
        self._space_creation_reward = 0.04  # Coefficient for creating space
        self._close_control_reward = 0.03  # Coefficient for keeping ball close
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._max_possession_steps = 0
        self._previous_distance_to_goal = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = None
        self._previous_player_position = None
        self._consecutive_possession_steps = 0
        self._max_possession_steps = 0
        self._previous_distance_to_goal = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self._previous_ball_position,
            'previous_player_position': self._previous_player_position,
            'consecutive_possession_steps': self._consecutive_possession_steps,
            'max_possession_steps': self._max_possession_steps,
            'previous_distance_to_goal': self._previous_distance_to_goal
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        checkpoint_state = from_pickle['CheckpointRewardWrapper']
        self._previous_ball_position = checkpoint_state['previous_ball_position']
        self._previous_player_position = checkpoint_state['previous_player_position']
        self._consecutive_possession_steps = checkpoint_state['consecutive_possession_steps']
        self._max_possession_steps = checkpoint_state['max_possession_steps']
        self._previous_distance_to_goal = checkpoint_state['previous_distance_to_goal']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_sequence_reward": [0.0] * len(reward),
            "ball_retention_reward": [0.0] * len(reward),
            "attacking_area_reward": [0.0] * len(reward),
            "penalty_area_reward": [0.0] * len(reward),
            "space_creation_reward": [0.0] * len(reward),
            "close_control_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # If goal is scored, give bonus for extended possession sequences
            if reward[rew_index] == 1:
                if self._max_possession_steps > 5:
                    components["dribbling_sequence_reward"][
                        rew_index] = self._dribbling_sequence_reward * self._max_possession_steps
                    reward[rew_index] += components["dribbling_sequence_reward"][rew_index]
                continue

            # Check if our team (left team, team 0) has possession
            has_possession = (o['ball_owned_team'] == 0 and
                              'ball_owned_player' in o and
                              o['ball_owned_player'] >= 0)

            if has_possession:
                active_player_idx = o['ball_owned_player']
                player_pos = o['left_team'][active_player_idx]
                ball_pos = o['ball'][:2]  # x, y coordinates only

                # Increment consecutive possession counter
                self._consecutive_possession_steps += 1
                self._max_possession_steps = max(self._max_possession_steps, self._consecutive_possession_steps)

                # Reward for extended dribbling sequences (continuous possession)
                if self._consecutive_possession_steps > 3:
                    dribbling_bonus = min(self._consecutive_possession_steps - 3, 10) * self._dribbling_sequence_reward
                    components["dribbling_sequence_reward"][rew_index] = dribbling_bonus
                    reward[rew_index] += dribbling_bonus

                # Reward for ball retention under pressure
                components["ball_retention_reward"][rew_index] = self._ball_retention_reward
                reward[rew_index] += self._ball_retention_reward

                # Reward for being in attacking third (opponent's half)
                if player_pos[0] > 0:
                    components["attacking_area_reward"][rew_index] = self._attacking_area_reward
                    reward[rew_index] += self._attacking_area_reward

                    # Extra reward for being in penalty area (close to goal)
                    distance_to_goal = ((player_pos[0] - 1.0) ** 2 + player_pos[1] ** 2) ** 0.5
                    if distance_to_goal < 0.3:  # Within penalty area
                        components["penalty_area_reward"][rew_index] = self._penalty_area_reward
                        reward[rew_index] += self._penalty_area_reward

                # Reward for close ball control (keeping ball near feet)
                ball_player_distance = ((ball_pos[0] - player_pos[0]) ** 2 +
                                        (ball_pos[1] - player_pos[1]) ** 2) ** 0.5
                if ball_player_distance < 0.05:  # Very close control
                    components["close_control_reward"][rew_index] = self._close_control_reward
                    reward[rew_index] += self._close_control_reward

                # Reward for space creation (moving towards goal while maintaining possession)
                if self._previous_player_position is not None and self._previous_distance_to_goal is not None:
                    current_distance_to_goal = ((player_pos[0] - 1.0) ** 2 + player_pos[1] ** 2) ** 0.5

                    # If player moved closer to goal while maintaining possession
                    if current_distance_to_goal < self._previous_distance_to_goal:
                        space_creation_bonus = (
                                                           self._previous_distance_to_goal - current_distance_to_goal) * self._space_creation_reward * 10
                        components["space_creation_reward"][rew_index] = space_creation_bonus
                        reward[rew_index] += space_creation_bonus

                    self._previous_distance_to_goal = current_distance_to_goal
                else:
                    self._previous_distance_to_goal = ((player_pos[0] - 1.0) ** 2 + player_pos[1] ** 2) ** 0.5

                # Update previous positions
                self._previous_ball_position = ball_pos.copy()
                self._previous_player_position = player_pos.copy()

            else:
                # Reset possession counter if ball is lost
                self._consecutive_possession_steps = 0
                self._previous_ball_position = None
                self._previous_player_position = None
                self._previous_distance_to_goal = None

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