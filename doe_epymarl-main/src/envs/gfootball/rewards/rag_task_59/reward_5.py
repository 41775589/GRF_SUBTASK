import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances rewards in a goalkeeper coordination task, including backup strategies 
    and efficient ball clearance to outfield players.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.clearance_checkpoints = {}  # Dict to store if a clearance to each player has already been rewarded
        self.goalkeeper_support_bonus = 0.5  # Extra reward for supporting the goalkeeper under pressure
        self.clearance_support_bonus = 0.1  # Reward for clearing the ball to another player
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.clearance_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.clearance_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        obs = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goalkeeper_support_reward": [0.0] * len(reward),
            "clearance_reward": [0.0] * len(reward)
        }

        for i, o in enumerate(obs):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and 'ball_owned_player' in o:
                # Strategy to support goalkeeper under pressure
                if obs[i]['left_team_roles'][o['ball_owned_player']] == 0:  # e_PlayerRole_GK
                    components["goalkeeper_support_reward"][i] = self.goalkeeper_support_bonus

                # Reward for clearing the ball to outfield players
                if o['game_mode'] in [3, 4]:  # Free kick or Corner kick scenarios
                    if 'right_team' in o:
                        distances = np.linalg.norm(
                            o['ball'][:2] - o['right_team'], axis=1)
                        min_idx = np.argmin(distances)
                        if min_idx not in self.clearance_checkpoints:
                            self.clearance_checkpoints[min_idx] = True
                            components["clearance_reward"][i] = self.clearance_support_bonus

            reward[i] += components["goalkeeper_support_reward"][i] + components["clearance_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
