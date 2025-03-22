import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for mastering offensive strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_recieved = 0.2
        self.shoot_accuracy = 1.0
        self.dribble_efficiency = 0.5
        self.player_position_advancement = 0.1

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_recieved": [0.0] * len(reward),
                      "shoot_accuracy": [0.0] * len(reward),
                      "dribble_efficiency": [0.0] * len(reward),
                      "player_position_advancement": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Include the reward for receiving passes
            if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:
                components["pass_recieved"][rew_index] = self.pass_recieved
                reward[rew_index] += components["pass_recieved"][rew_index]

            # Include additional reward for accurate shoots toward the opponent's goal
            if o['game_mode'] == 6:  # Assuming game mode 6 is for shooting/scoring
                components["shoot_accuracy"][rew_index] = self.shoot_accuracy
                reward[rew_index] += components["shoot_accuracy"][rew_index]

            # Dribbling effectiveness checked by distance covered while possessing the ball
            if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Assuming index 9 is dribble action
                components["dribble_efficiency"][rew_index] = self.dribble_efficiency
                reward[rew_index] += components["dribble_efficiency"][rew_index]

            # Encourage forward movement towards the opponent's goal
            if o['ball_owned_player'] == o['active'] and o['ball'][0] > 0:  # Assuming positive x is toward opponent's goal
                progress = o['ball'][0] - (-1)  # from -1 (own goal) to +1 (opponent's goal)
                components["player_position_advancement"][rew_index] += self.player_position_advancement * progress
                reward[rew_index] += components["player_position_advancement"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
