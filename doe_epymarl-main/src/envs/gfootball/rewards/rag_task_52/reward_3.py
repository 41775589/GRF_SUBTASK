import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to enhance defending strategies by focusing on tackling proficiency,
    efficient movement control, and pressured passing tactics.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define custom parameters for reward scaling
        self.tackle_reward_coefficient = 0.05
        self.movement_control_coefficient = 0.03
        self.passing_pressure_coefficient = 0.07

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "movement_control_reward": [0.0] * len(reward),
            "passing_pressure_reward": [0.0] * len(reward)
        }

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Tackling reward: Increase whenever the opponent loses ball control
            if o['ball_owned_team'] == 1:  # Assuming 0 is left (own), 1 is right (opponent)
                components["tackle_reward"][rew_index] = self.tackle_reward_coefficient
                reward[rew_index] += components["tackle_reward"][rew_index]
            
            # Movement control reward: Active if opposing team's velocity near the ball is low
            if o['ball_owned_team'] == 1 and np.linalg.norm(o['right_team_direction'][o['ball_owned_player']]) < 0.01:
                components["movement_control_reward"][rew_index] = self.movement_control_coefficient
                reward[rew_index] += components["movement_control_reward"][rew_index]
            
            # Passing pressure reward: Active if close to the ball when the opponent has possession
            if o['ball_owned_team'] == 1:
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][:2]
                distance_to_ball = np.linalg.norm(player_pos - ball_pos)
                if distance_to_ball < 0.1:
                    components["passing_pressure_reward"][rew_index] = self.passing_pressure_coefficient
                    reward[rew_index] += components["passing_pressure_reward"][rew_index]

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
