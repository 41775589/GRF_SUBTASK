import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a strategic positioning and passing reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positioning_coefficient = 0.1
        self.passing_coefficient = 0.5
        self.defensive_form_coefficient = 0.3

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
        reward_components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, reward_components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            obs = observation[idx]
            ball_pos = obs.get('ball', [0, 0, 0])
            player_pos = obs['right_team'] if obs['ball_owned_team'] == 1 else obs['left_team'][obs.get('active')]

            # Evaluate defensive positioning
            def_distance_to_ball = np.linalg.norm(np.array(player_pos[:2]) - np.array(ball_pos[:2]))
            defensive_reward = np.exp(-def_distance_to_ball) * self.defensive_form_coefficient
            
            # Check for passing actions
            if obs.get('game_mode') in [2, 3, 4]:  # FreeKick, GoalKick, Corner
                passing_reward = self.passing_coefficient * (1 - np.min([def_distance_to_ball, 1]))
                reward_components["passing_reward"][idx] = passing_reward

            reward_components["positioning_reward"][idx] = defensive_reward
            reward[idx] += reward_components["base_score_reward"][idx] + reward_components["positioning_reward"][idx] + reward_components["passing_reward"][idx]

        return reward, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
