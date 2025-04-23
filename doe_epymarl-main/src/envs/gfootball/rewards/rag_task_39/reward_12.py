import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Reward wrapper to focus on mastering the clearance of the ball from defensive zones under pressure."""

    def __init__(self, env):
        super().__init__(env)
        # Counts how often sticky actions are used
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Threshold for "safe zone" beyond which clearance is considered effective
        self.safe_zone_threshold = 0.5
        self.clearance_attempt_weight = 0.1
        self.clearance_success_weight = 0.3
        self.under_pressure_penalty = -0.05
        self.successful_clearance_bonus = 0.5
        self.past_ball_positions = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.past_ball_positions = []
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['past_ball_positions'] = self.past_ball_positions
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.past_ball_positions = from_pickle['past_ball_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        new_rewards = []

        for o, rew in zip(observation, reward):
            # Identify ball owned by the defensive team (team 0 is usually the controlled/learning agent's team)
            is_ball_owned_our_team = o['ball_owned_team'] == 0
            ball_in_defensive_half = o['ball'][0] < 0  # Assuming left-half is defensive
            player_defensive_position = (o['left_team'][o['active']][0] < self.safe_zone_threshold)
            under_pressure = (o['right_team'][:, 0] - o['left_team'][o['active']][0]).min() < 0.1
            
            components.setdefault("clearance_attempt", []).append(0.0)
            components.setdefault("clearance_success", []).append(0.0)
            components.setdefault("under_pressure_penalty", []).append(0.0)

            clearance_attempt_rew = 0
            clearance_success_rew = 0
            under_pressure_pen = 0

            if is_ball_owned_our_team and player_defensive_position:
                components["clearance_attempt"][-1] = self.clearance_attempt_weight
                clearance_attempt_rew = self.clearance_attempt_weight

                if ball_in_defensive_half:
                    self.past_ball_positions.append(o['ball'][0])
                    if self.past_ball_positions and o['ball'][0] > max(self.past_ball_positions):
                        components["clearance_success"][-1] = self.clearance_success_weight
                        clearance_success_rew = self.successful_clearance_bonus
                    
                    if under_pressure:
                        components["under_pressure_penalty"][-1] = self.under_pressure_penalty
                        under_pressure_pen = self.under_pressure_penalty

            new_rew = rew + clearance_attempt_rew + clearance_success_rew + under_pressure_pen
            new_rewards.append(new_rew)

        return new_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
