import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym Reward Wrapper tailored to encourage rapid changes in sprinting states, 
    rewarding both sustained sprinting for quick movements and effective stops to enhance agility.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sprint_active_duration = 0
        self.stop_sprint_rewards = 0
        self.action_last = None

    def reset(self):
        self.sprint_active_duration = 0
        self.stop_sprint_rewards = 0
        self.action_last = None
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sprint_reward": [0.0],
            "stop_sprint_reward": [0.0]
        }
        
        if observation is None:
            return reward, components
       
        o = observation[0]  # Single-agent environment
        sprint_action_current = o['sticky_actions'][8]  # Sprint action index
        
        if sprint_action_current:
            if self.action_last != sprint_action_current:  # Transition to sprinting
                self.sprint_active_duration = 1
            else:
                self.sprint_active_duration += 1
            components["sprint_reward"][0] = 0.05 * self.sprint_active_duration  # Reward grows with duration of sprint
        else:
            if self.action_last == 1:  # Was sprinting but now stopped
                components["stop_sprint_reward"][0] = 0.1  # Reward sprint cessation
                self.stop_sprint_rewards += 1
        
        self.action_last = sprint_action_current
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
        to_pickle['sprint_active_duration'] = self.sprint_active_duration
        to_pickle['stop_sprint_rewards'] = self.stop_sprint_rewards
        to_pickle['action_last'] = self.action_last
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_active_duration = from_pickle['sprint_active_duration']
        self.stop_sprint_rewards = from_pickle['stop_sprint_rewards']
        self.action_last = from_pickle['action_last']
        return from_pickle
