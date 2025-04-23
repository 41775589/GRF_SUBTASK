import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward system focusing on transition skills:
    Short Pass, Long Pass, and Dribble under pressure.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_bonus = 0.3
        self.control_keeping_bonus = 0.1
        self.dribble_advancement_bonus = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_bonus": [0.0] * len(reward),
                      "control_bonus": [0.0] * len(reward),
                      "dribble_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            active_player = o['active']
            own_team = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
            control = o['ball_owned_team'] == o['game_mode']

            # Calculate control keeping bonus
            if control:
                components["control_bonus"][idx] = self.control_keeping_bonus
                reward[idx] += components["control_bonus"][idx]

            # Calculate pass completion bonus
            if o['game_mode'] in [2, 3, 4, 5]:  # modes related to pass/start/stop
                components["pass_bonus"][idx] = self.pass_completion_bonus
                reward[idx] += components["pass_bonus"][idx]

            # Calculate dribbling bonus, encouraging keeping the ball under pressure
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Dribbling active
                dribble_bonus_factor = np.linalg.norm(o['ball_direction'][0:2])
                components["dribble_bonus"][idx] = self.dribble_advancement_bonus * dribble_bonus_factor
                reward[idx] += components["dribble_bonus"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
