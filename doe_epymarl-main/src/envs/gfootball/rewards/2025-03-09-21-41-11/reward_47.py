import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds structured reward components designed to enhance offensive gameplay training."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.dribbling_bonus = 0.2
        self.pass_bonus = 0.1
        self.shooting_bonus = 0.3

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'dribbling_bonus': self.dribbling_bonus,
            'pass_bonus': self.pass_bonus,
            'shooting_bonus': self.shooting_bonus
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        pik = from_pickle['CheckpointRewardWrapper']
        self.dribbling_bonus = pik['dribbling_bonus']
        self.pass_bonus = pik['pass_bonus']
        self.shooting_bonus = pik['shooting_bonus']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "dribbling_reward": [0.0] * len(reward),
            "pass_reward": [0.0] * len(reward),
            "shooting_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Reward dribbling: active player in possession of the ball and not stationary
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player'] and np.any(o['ball_direction'] != [0, 0, 0]):
                components["dribbling_reward"][i] = self.dribbling_bonus

            # Reward passing: ball change control to a teammate without a loss of ball to opponent
            if o['game_mode'] in [2, 5, 6]:  # Game modes related to ball control changes
                if o['ball_owned_team'] == 0:  # Our team has ball after a pass
                    components["pass_reward"][i] = self.pass_bonus

            # Reward shooting: shots directed towards the opponent's goal
            if np.linalg.norm(o['ball_direction'][:2] - np.array([1, 0])) < np.linalg.norm(o['ball_direction'][:2]):
                components["shooting_reward"][i] = self.shooting_bonus

            reward[i] += (components["dribbling_reward"][i] + components["pass_reward"][i] + components["shooting_reward"][i])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, values in components.items():
            info[f"component_{key}"] = sum(values)

        return observation, reward, done, info
