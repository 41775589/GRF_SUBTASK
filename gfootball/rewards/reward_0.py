import gym
import numpy as np


class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards high-speed ball carrying and breakthrough play."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._previous_ball_position = None
        self._previous_player_position = None
        self._previous_opponents_beaten = 0
        self._ball_carry_distance = 0.0
        self._consecutive_possession_steps = 0

        # Reward coefficients for easy tuning
        self._ball_carry_reward_coeff = 2.0
        self._sprint_carry_bonus_coeff = 1.5
        self._opponent_beaten_reward_coeff = 0.3
        self._penalty_area_bonus_coeff = 0.5
        self._possession_momentum_coeff = 0.02

        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._previous_ball_position = None
        self._previous_player_position = None
        self._previous_opponents_beaten = 0
        self._ball_carry_distance = 0.0
        self._consecutive_possession_steps = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self._previous_ball_position,
            'previous_player_position': self._previous_player_position,
            'previous_opponents_beaten': self._previous_opponents_beaten,
            'ball_carry_distance': self._ball_carry_distance,
            'consecutive_possession_steps': self._consecutive_possession_steps
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        wrapper_state = from_pickle['CheckpointRewardWrapper']
        self._previous_ball_position = wrapper_state['previous_ball_position']
        self._previous_player_position = wrapper_state['previous_player_position']
        self._previous_opponents_beaten = wrapper_state['previous_opponents_beaten']
        self._ball_carry_distance = wrapper_state['ball_carry_distance']
        self._consecutive_possession_steps = wrapper_state['consecutive_possession_steps']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_carry_reward": [0.0] * len(reward),
            "sprint_carry_bonus": [0.0] * len(reward),
            "opponent_beaten_reward": [0.0] * len(reward),
            "penalty_area_bonus": [0.0] * len(reward),
            "possession_momentum": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Start with base reward
            total_reward = components["base_score_reward"][rew_index]

            # Check if our team (left team, team 0) has ball possession
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0 and
                    'ball_owned_player' in o and o['ball_owned_player'] == o['active']):

                current_ball_pos = o['ball'][:2]  # [x, y] position
                current_player_pos = o['left_team'][o['active']]  # Active player position

                # Increment consecutive possession counter
                self._consecutive_possession_steps += 1

                # Calculate ball carrying distance (forward progress toward opponent goal)
                if self._previous_ball_position is not None:
                    # Forward progress is increase in x-coordinate (toward right goal at x=1)
                    forward_progress = current_ball_pos[0] - self._previous_ball_position[0]
                    if forward_progress > 0:  # Only reward forward movement
                        carry_distance = forward_progress
                        self._ball_carry_distance += carry_distance

                        components["ball_carry_reward"][rew_index] = carry_distance * self._ball_carry_reward_coeff
                        total_reward += components["ball_carry_reward"][rew_index]

                # Sprint carrying bonus - reward when sprinting while carrying ball
                is_sprinting = o['sticky_actions'][8] == 1  # sprint action is at index 8
                if is_sprinting and self._previous_ball_position is not None:
                    sprint_bonus = 0.1
                    components["sprint_carry_bonus"][rew_index] = sprint_bonus * self._sprint_carry_bonus_coeff
                    total_reward += components["sprint_carry_bonus"][rew_index]

                # Count opponents beaten (opponents behind the ball carrier)
                current_opponents_beaten = 0
                player_x = current_player_pos[0]
                for opponent_pos in o['right_team']:
                    if opponent_pos[0] < player_x:  # Opponent is behind us
                        current_opponents_beaten += 1

                # Reward for beating more opponents than before
                if current_opponents_beaten > self._previous_opponents_beaten:
                    opponents_newly_beaten = current_opponents_beaten - self._previous_opponents_beaten
                    components["opponent_beaten_reward"][
                        rew_index] = opponents_newly_beaten * self._opponent_beaten_reward_coeff
                    total_reward += components["opponent_beaten_reward"][rew_index]

                self._previous_opponents_beaten = current_opponents_beaten

                # Penalty area bonus - extra reward for carrying ball into penalty area
                # Right penalty area is approximately x > 0.8 and |y| < 0.2
                if current_ball_pos[0] > 0.8 and abs(current_ball_pos[1]) < 0.2:
                    components["penalty_area_bonus"][rew_index] = self._penalty_area_bonus_coeff
                    total_reward += components["penalty_area_bonus"][rew_index]

                # Possession momentum - small continuous reward for maintaining possession
                momentum_reward = min(self._consecutive_possession_steps * 0.001, 0.05)
                components["possession_momentum"][rew_index] = momentum_reward * self._possession_momentum_coeff
                total_reward += components["possession_momentum"][rew_index]

                # Update previous positions
                self._previous_ball_position = current_ball_pos.copy()
                self._previous_player_position = current_player_pos.copy()

            else:
                # Lost possession, reset counters
                self._consecutive_possession_steps = 0
                self._previous_opponents_beaten = 0
                if self._previous_ball_position is not None:
                    self._previous_ball_position = o['ball'][:2].copy()

            reward[rew_index] = total_reward

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