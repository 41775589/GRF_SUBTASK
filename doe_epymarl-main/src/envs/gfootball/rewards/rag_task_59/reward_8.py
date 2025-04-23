import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specific to goalkeeper coordination, focusing on backup strategies and
    effective ball clearing in high-pressure situations."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_positions = {}
        self._backup_reward = 0.1
        self._clearing_reward = 0.25

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.goalkeeper_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self.goalkeeper_positions = from_picle['CheckpointRewardWrapper']
        return from_picle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "backup_reward": [0.0] * len(reward),
                      "clearing_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0:  # Team of agent
                goalie_index = np.where(o['left_team_roles'] == 0)[0][0]  # Assuming 0 is GK
                goalie_pos = o['left_team'][goalie_index]
                ball_pos = o['ball'][:2]  # ignore z-coordinate
                distance_to_goalie = np.linalg.norm(goalie_pos - ball_pos)

                if distance_to_goalie < 0.1:
                    self.goalkeeper_positions[rew_index] = goalie_pos
                    components["backup_reward"][rew_index] = self._backup_reward
                    reward[rew_index] += self._backup_reward

                # Assess clearing rewards
                if ('ball_owned_player' in o and
                    o['ball_owned_player'] == goalie_index and 
                    np.any(np.abs(o['ball_direction'][:2]) > 1)):  # Strong clear
                    components["clearing_reward"][rew_index] = self._clearing_reward
                    reward[rew_index] += self._clearing_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
