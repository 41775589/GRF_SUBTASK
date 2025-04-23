import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward function focusing on offensive strategies:
    mastering accurate shooting, effective dribbling to evade opponents,
    and practicing different pass types to break defensive lines.
    """
    def __init__(self, env):
        super().__init__(env)
        self.shooting_reward = 1.0
        self.dribbling_reward = 0.5
        self.passing_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Reward for successful shots on goal
                if o['game_mode'] == 6 and 'goal' in o['designated']:  # Game mode 6 is Penalty
                    components["shooting_reward"][i] = self.shooting_reward
                    reward[i] += components["shooting_reward"][i]

                # Reward for dribbling (maintaining ball possession amidst opponents)
                dribble_action_active = o['sticky_actions'][9]  # index 9 is dribble action
                if dribble_action_active:
                    components["dribbling_reward"][i] = self.dribbling_reward
                    reward[i] += components["dribbling_reward"][i]

                # Reward for effective passing (high and long passes)
                high_or_long_pass = 'high_pass' in o['designated'] or 'long_pass' in o['designated']
                if high_or_long_pass:
                    components["passing_reward"][i] = self.passing_reward
                    reward[i] += components["passing_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        updated_reward, components = self.reward(reward)
        info["final_reward"] = sum(updated_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, updated_reward, done, info
