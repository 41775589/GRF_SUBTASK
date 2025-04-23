import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that emphasizes wide midfield play, including successful
    high passes and effective wide field positioning to stretch the defense.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_quality_threshold = 0.7  # Threshold for considering a high pass effective
        self.high_pass_reward = 0.5
        self.positioning_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for effective high passing
            if o['sticky_actions'][9] == 1:  # Index 9 corresponds to action_dribble, assuming it as the high pass action
                ball_speed = np.linalg.norm(o['ball_direction'][0:2])
                if ball_speed > self.pass_quality_threshold:
                    components["high_pass_reward"][rew_index] = self.high_pass_reward
                    reward[rew_index] += components["high_pass_reward"][rew_index]

            # Reward for effective positioning to stretch opposition's defense
            # We estimate this by the x-position and lateral movement (y-direction)
            player_x, player_y = o['left_team'][o['active']]
            if abs(player_y) > 0.3:  # Assume a y-value threshold that defines 'wide' positioning
                movement_y = abs(o['left_team_direction'][o['active']][1])
                components["positioning_reward"][rew_index] = self.positioning_reward * movement_y
                reward[rew_index] += components["positioning_reward"][rew_index]
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
