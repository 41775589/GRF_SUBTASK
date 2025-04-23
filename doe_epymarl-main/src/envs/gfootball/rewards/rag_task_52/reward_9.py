import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards to train agents on defensive strategies, including tackling proficiency,
    efficient movement control, and pressured passing tactics."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.2
        self.pressure_pass_reward = 0.3
        self.efficient_movement_reward = 0.1

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
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "pressure_pass_reward": [0.0] * len(reward),
                      "efficient_movement_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, o in enumerate(observation):
            # Detect efficient movement control (e.g., sudden stops against opponent's direction)
            player_dir = o.get("active", {}).get("direction")
            opponent_dir = o.get("opposition", {}).get("direction")
            if player_dir and opponent_dir and np.dot(player_dir, opponent_dir) < 0:
                components["efficient_movement_reward"][i] = self.efficient_movement_reward
                reward[i] += components["efficient_movement_reward"][i]

            # Reward for successful tackles
            if o['game_mode'] == 5 and o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                components["tackle_reward"][i] = self.tackle_reward
                reward[i] += components["tackle_reward"][i]

            # Reward for making a successful pass under pressure
            if o['game_mode'] in [2, 3] and o['ball_owned_team'] == 0 and o['sticky_actions'][9]:  # Assuming index 9 is 'pass'
                components["pressure_pass_reward"][i] = self.pressure_pass_reward
                reward[i] += components["pressure_pass_reward"][i]

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
                self.sticky_actions_counter[i] = action
                info[f"sticky_actions_{i}"] = agent_obs['sticky_actions'][i]
        return observation, reward, done, info
