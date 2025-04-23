import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that augments the reward for key defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.num_successful_tackles = {}
        self.reward_for_tackle = 1.0
        self.penalty_for_foul = -0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.num_successful_tackles = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.num_successful_tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.num_successful_tackles = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "foul_penalty": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if obs['game_mode'] == 6:  # Game mode is penalty
                continue  # Avoid modifying tackle rewards during penalties

            # Check for tackles
            if obs['sticky_actions'][9]:  # action_dribble index for tackles
                self.sticky_actions_counter[9] += 1
                if self.sticky_actions_counter[9] == 1:
                    components["tackle_reward"][i] = self.reward_for_tackle
                    reward[i] += components["tackle_reward"][i]

            # Check for fouls
            if obs['left_team_yellow_card'][obs['active']] or obs['right_team_yellow_card'][obs['active']]:
                components["foul_penalty"][i] = self.penalty_for_foul
                reward[i] += components["foul_penalty"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
