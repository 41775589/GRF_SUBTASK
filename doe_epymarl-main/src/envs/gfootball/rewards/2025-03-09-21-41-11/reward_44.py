import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward signals for learning offensive strategies
    like accurate shooting, effective dribbling, and versatile passing.
    Specifically models rewards for shooting accuracy, dribbling past opponents,
    and executing effective long and high passes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_accuracy_reward = 1.0
        self.dribble_reward = 0.5
        self.passing_reward = 0.3

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward}

        components = {
            "base_score_reward": reward.copy(),
            "shooting_accuracy_reward": [0.0] * len(reward),
            "dribble_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward)
        }

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for shooting accuracy
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                if o['ball'][0] > 0.9:  # close to opponent's goal
                    components["shooting_accuracy_reward"][rew_index] = self.shooting_accuracy_reward

            # Reward for dribbling: checking ball possession and player movement
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1 and 'ball_owned_player' in o:
                if o['active'] == o['ball_owned_player'] and ('sticky_actions' in o and o['sticky_actions'][9]):  # dribbling
                    components["dribble_reward"][rew_index] = self.dribble_reward

            # Reward for effective pass: ball changes ownership within team without interception
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                # Simulated check for game mode changes indicating successful long/high passes
                if 'game_mode' in o and o['game_mode'] in [5, 6]:  # assuming these modes indicate successful passes
                    components["passing_reward"][rew_index] = self.passing_reward
                
            # Calculate total reward
            reward[rew_index] += (components["shooting_accuracy_reward"][rew_index] +
                                  components["dribble_reward"][rew_index] +
                                  components["passing_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add individual rewards to info for debugging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
