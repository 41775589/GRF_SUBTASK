import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that enhances sprint and stopping sprint techniques for agility improvement.
    It focuses on rewarding rapidly alternating between sprinting and non-sprinting states to improve 
    agility as per the specified training goal.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sprint_action_duration = 0
        self.sprint_stop_action_duration = 0

    def reset(self):
        self.sprint_action_duration = 0
        self.sprint_stop_action_duration = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "sprint_reward": [0.0], "stop_sprint_reward": [0.0]}

        if observation is None:
            return reward, components

        o = observation[0]  # Single-agent case
        sprint_active = o['sticky_actions'][8]  # Index 8 corresponds to sprint action

        if sprint_active:
            # Reward continuous sprinting
            self.sprint_action_duration += 1
            if self.sprint_action_duration > 1:  # Encourage sustained sprinting
                components["sprint_reward"][0] = 0.05 * self.sprint_action_duration
        else:
            if self.sprint_action_duration > 0:
                # Reward for stopping sprint after a sprint period
                components["stop_sprint_reward"][0] = 0.1
                self.sprint_stop_action_duration += 1
            self.sprint_action_duration = 0

        # Accumulate total reward
        reward[0] += components["sprint_reward"][0] + components["stop_sprint_reward"][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['sprint_action_duration'] = self.sprint_action_duration
        to_pickle['sprint_stop_action_duration'] = self.sprint_stop_action_duration
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_action_duration = from_pickle['sprint_action_duration']
        self.sprint_stop_action_duration = from_pickle['sprint_stop_action_duration']
        return from_pickle
