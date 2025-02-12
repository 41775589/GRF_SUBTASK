import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a targeted reward focused on midfielder/defender tasks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize the specific metrics that will be important for this agent's performance.
        self.pass_accuracy = 0
        self.dribble_effectiveness = 0
        self.defensive_actions = 0

    def reset(self):
        # Reset all internally tracked measures.
        self.pass_accuracy = 0
        self.dribble_effectiveness = 0
        self.defensive_actions = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        modified_reward = reward.copy()
        components = {
            "base_score_reward": reward.copy(),
            "pass_accuracy_reward": [0.0],
            "dribble_effectiveness_reward": [0.0],
            "defensive_actions_reward": [0.0]
        }

        if observation is None:
            return reward, components

        o = observation[0]
        # Rewards for proficient passing (High and Long passes)
        if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
            if o['game_mode'] in [4, 5]:  # High or long pass modes
                self.pass_accuracy += 1
                components["pass_accuracy_reward"][0] = 0.1

        # Rewards for effective dribbling
        if 'sticky_actions' in o and o['sticky_actions'][9]:  # index 9 corresponds to dribble in actions
            self.dribble_effectiveness += 1
            components["dribble_effectiveness_reward"][0] = 0.1

        # Rewards for defensive actions (e.g., tackles, interceptions)
        if 'game_mode' in o and o['game_mode'] == 7:  # Defensive action modes
            self.defensive_actions += 1
            components["defensive_actions_reward"][0] = 0.1

        # Adding all additional rewards to the base reward
        modified_reward[0] += (components["pass_accuracy_reward"][0] +
                               components["dribble_effectiveness_reward"][0] +
                               components["defensive_actions_reward"][0])

        return modified_reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value pair into info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
