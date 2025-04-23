import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for decision-making in shooting and passing scenarios in a soccer game simulation."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_accuracy_reward = 0.2
        self.passing_accuracy_reward = 0.1
        self.positive_scenarios = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positive_scenarios = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'positive_scenarios': self.positive_scenarios}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.positive_scenarios = from_pickle['CheckpointRewardWrapper']['positive_scenarios']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_accuracy_reward": [0.0] * len(reward),
            "passing_accuracy_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            if 'ball_owned_team' in o and o['ball_owned_team'] in [0, 1] and o['game_mode'] == 0:
                ball_owner = o['ball_owned_player']
                if ball_owner == o['active']:
                    shooting_condition = any([action for action in o['sticky_actions'][6:8]])
                    passing_condition = o['sticky_actions'][8] == 1

                    if shooting_condition and self.positive_scenarios.get(i, {}).get('shooting_accuracy', False) == False:
                        components['shooting_accuracy_reward'][i] = self.shooting_accuracy_reward
                        self.positive_scenarios.setdefault(i, {})['shooting_accuracy'] = True

                    if passing_condition and self.positive_scenarios.get(i, {}).get('passing_accuracy', False) == False:
                        components['passing_accuracy_reward'][i] = self.passing_accuracy_reward
                        self.positive_scenarios.setdefault(i, {})['passing_accuracy'] = True

        final_rewards = [
            components['base_score_reward'][i] +
            components['shooting_accuracy_reward'][i] +
            components['passing_accuracy_reward'][i]
            for i in range(len(reward))
        ]

        return final_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
