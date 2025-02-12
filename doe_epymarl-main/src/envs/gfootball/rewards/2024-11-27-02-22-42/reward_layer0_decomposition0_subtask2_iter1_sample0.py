import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.pass_success = 0
        self.defensive_actions = 0
        self.possession_time = 0

    def reset(self):
        # Reset counters for each aspect we are focusing on
        self.pass_success = 0
        self.defensive_actions = 0
        self.possession_time = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save the states related to our customized reward components
        to_pickle['pass_success'] = self.pass_success
        to_pickle['defensive_actions'] = self.defensive_actions
        to_pickle['possession_time'] = self.possession_time
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Assign states to our parameters from the saved versions
        self.env.set_state(state)
        from_pickle = self.env.get_state(state)
        self.pass_success = from_pickle['pass_success']
        self.defensive_actions = from_pickle['defensive_actions']
        self.possession_time = from_pickle['possession_time']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        modified_reward = np.copy(reward)
        components = {
            "base_score_reward": reward.copy(),  # Score before modifications
            "pass_accuracy_reward": [0.0],
            "defensive_actions_reward": [0.0],
            "possession_time_reward": [0.0]
        }

        o = observation[0]  # Assume a single agent observation in the environment

        # Encourage passing - Both successful high & long passes
        if 'sticky_actions' in o and (o['sticky_actions'][4] or o['sticky_actions'][5]):  # LongPass, HighPass
            self.pass_success += 1
            components["pass_accuracy_reward"][0] = 0.1

        # Encourage defensive maneuvers - Improved positioning and interception logic
        if 'game_mode' in o and o['game_mode'] in [3, 4]:  # Defensive modes
            self.defensive_actions += 1
            components["defensive_actions_reward"][0] = 0.2

        # Encourage maintaining possession under pressure
        if 'ball_owned_team' in o and o['ball_owned_team'] == o['active']:
            self.possession_time += 1
            components["possession_time_reward"][0] = 0.05

        # Sum all contributions
        for key in components:
            if key != "base_score_reward":
                modified_reward += components[key]

        return modified_reward.tolist(), components

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
