import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym Reward Wrapper to encourage agile movements by increasing or 
    decreasing the frequency of sprinting along with its cessation, which 
    should help in improving rapid change of pace and swift stops.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sprint_active_duration = 0
        self.stop_duration = 0

    def reset(self):
        self.sprint_active_duration = 0
        self.stop_duration = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sprint_reward": [0.0],
            "sprint_stop_reward": [0.0]
        }

        if observation is None:
            return reward, components

        o = observation[0]  # Assuming single-agent environment
        sprint_active = o['sticky_actions'][8]  # Sprint action index

        if sprint_active:
            self.sprint_active_duration += 1
            components["sprint_reward"][0] = 0.02 * self.sprint_active_duration
        else:
            if self.sprint_active_duration > 0:
                # Reward the stop but only if it follows a sprint
                components["sprint_stop_reward"][0] = 0.1
                self.stop_duration += 1
            self.sprint_active_duration = 0

        reward[0] += components["sprint_reward"][0] + components["sprint_stop_reward"][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['sprint_active_duration'] = self.sprint_active_duration
        to_pickle['stop_duration'] = self.stop_duration
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_active_duration = from_pickle['sprint_active_duration']
        self.stop_duration = from_pickle['stop_duration']
        return from_pickle
