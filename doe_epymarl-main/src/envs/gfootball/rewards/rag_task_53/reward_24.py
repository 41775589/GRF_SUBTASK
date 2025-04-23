import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for maintaining control and making strategic plays."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.safe_play_zones = [0.2, 0.4, 0.6, 0.8]  # Defines the distance thresholds on the field for rewards
        self.control_reward_factor = 0.05  # Reward for maintaining control under pressure
        self.passing_reward_factor = 0.1   # Reward for successful, strategic passes

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "control_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            if obs['ball_owned_team'] == obs['active']:
                # Check distance to goal and provide rewards for maintaining position in safe play zones
                if any([abs(obs['ball'][0]) > threshold for threshold in self.safe_play_zones]):
                    components["control_reward"][idx] = self.control_reward_factor
                    reward[idx] += components["control_reward"][idx]
                
                # Check passing efficacy (if a pass is made under pressure)
                if obs.get('game_mode') == 3:  # Assuming game mode 3 relates to successful strategic play
                    components["passing_reward"][idx] = self.passing_reward_factor
                    reward[idx] += components["passing_reward"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        info.update({'sticky_actions': self.sticky_actions_counter})
        return observation, reward, done, info
