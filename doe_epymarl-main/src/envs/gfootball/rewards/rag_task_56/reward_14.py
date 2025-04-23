import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to enhance defensive skills, focusing on goalkeeper shot-stopping 
    and defender tackling and ball retention.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_shot_stopping_reward = 0.2
        self.defender_tackling_reward = 0.1
        self.ball_retention_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # As there is no specific state needed to store, we simply return it.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goalkeeper_shot_stopping": [0.0] * len(reward),
            "defender_tackling": [0.0] * len(reward),
            "ball_retention": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Reward for goalkeeper successfully stopping a shot
            if o['right_team_roles'][o['active']] == 0 and o['ball_owned_team'] == 1:  # Assuming role 0 is the goalkeeper
                components["goalkeeper_shot_stopping"][i] += self.goalkeeper_shot_stopping_reward

            # Defender tackling and ball retention
            if o['ball_owned_team'] == 0 and o['right_team_roles'][o['active']] in [1, 2, 3, 4]:  # Assuming roles 1-4 are defenders
                components["defender_tackling"][i] += self.defender_tackling_reward
                if np.any(o['sticky_actions'][8]):  # Assuming action 8 is related to maintaining possession
                    components["ball_retention"][i] += self.ball_retention_reward

            # Update the total reward with additional components
            reward[i] += (components["goalkeeper_shot_stopping"][i] +
                          components["defender_tackling"][i] +
                          components["ball_retention"][i])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
