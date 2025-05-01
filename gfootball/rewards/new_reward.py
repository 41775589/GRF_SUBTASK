import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the reward function by focusing on quick defensive transitions between moving and 
    stopping states, significantly highlighting the importance of stopping ball progression by the opponent.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dynamic_stopping_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for i, o in enumerate(observation):
            if o is None:
                continue
            
            # Analyze player and ball dynamics to enhance defensive capabilities
            player_speed = np.linalg.norm(o['left_team_direction'][o['active']])
            ball_approaching = np.linalg.norm(o['ball_direction']) > 0.01 and o['ball_owned_team'] == 1

            # Encourage rapid stopping if the ball is approaching from the opponent
            if player_speed < 0.02 and ball_approaching:
                components["dynamic_stopping_reward"][i] = 0.1
                reward[i] += components["dynamic_stopping_reward"][i]
            
            # Calculate extra reward for strategic positioning when the team does not possess the ball
            ball_owned_by_opponent = (o['ball_owned_team'] == 1)
            strategic_defensive_position = abs(o['left_team'][o['active']][0]) < 0.2
            if ball_owned_by_opponent and strategic_defensive_position:
                components["dynamic_stopping_reward"][i] += 0.2
                reward[i] += components["dynamic_stopping_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["total_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
