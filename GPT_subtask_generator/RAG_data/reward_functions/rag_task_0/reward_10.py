import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incorporates rewards focused on offensive skills like
    shooting accuracy, dribbling, and passing to break defensive lines."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_accuracy_reward = 0.5
        self.dribbling_skill_reward = 0.3
        self.passing_accuracy_reward = 0.2

    def reset(self):
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Restore any internal state from the pickle if needed
        return from_pickle

    def reward(self, reward):
        """Modify reward based on offensive gameplay skills:
        - Shooting accuracy
        - Effective dribbling
        - Successful long and high passes"""

        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_accuracy_reward": [0.0] * len(reward),
            "dribbling_skill_reward": [0.0] * len(reward),
            "passing_accuracy_reward": [0.0] * len(reward),
        }

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            if o['game_mode'] == 6:  # Penalty mode
                if o['score'][0] > o['score'][1]:  # Check for a successful goal
                    components["shooting_accuracy_reward"][i] += self.shooting_accuracy_reward
                    reward[i] += self.shooting_accuracy_reward

            if o['sticky_actions'][9] == 1:  # Dribble action is active
                components["dribbling_skill_reward"][i] += self.dribbling_skill_reward
                reward[i] += self.dribbling_skill_reward

            # Evaluating effective passing
            if o['game_mode'] in [3, 5]:  # Free kick or Throw-in mode
                components["passing_accuracy_reward"][i] += self.passing_accuracy_reward
                reward[i] += self.passing_accuracy_reward

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
