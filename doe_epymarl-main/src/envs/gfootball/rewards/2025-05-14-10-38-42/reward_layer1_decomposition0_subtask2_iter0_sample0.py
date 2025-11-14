import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to enhance sprint and stop-sprint techniques for agility improvement."""
    def __init__(self, env):
        super().__init__(env)
        # Initialize sprint action counter
        self.sprint_action_counter = 0
        # Initialize counter for stopping sprint
        self.stop_sprint_action_counter = 0

    def reset(self):
        # Reset action counters when environment is reset
        self.sprint_action_counter = 0
        self.stop_sprint_action_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        # Store the current state of the action counters in the pickle
        to_pickle['sprint_action_counter'] = self.sprint_action_counter
        to_pickle['stop_sprint_action_counter'] = self.stop_sprint_action_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Retrieve the action counters state from the pickle
        from_pickle = self.env.set_state(state)
        self.sprint_action_counter = from_pickle['sprint_action_counter']
        self.stop_sprint_action_counter = from_pickle['stop_sprint_action_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "sprint_bonus": [0.0], "stop_sprint_bonus": [0.0]}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sprint_active = o['sticky_actions'][8]  # index 8 corresponds to sprint action
            if sprint_active:
                components["sprint_bonus"][rew_index] = 0.05  # gain a small bonus for sprinting
                self.sprint_action_counter += 1
            stop_sprint_active = (not sprint_active) and (self.sprint_action_counter > 0)
            if stop_sprint_active:
                components["stop_sprint_bonus"][rew_index] = 0.05  # gain a bonus for stopping sprint
                self.stop_sprint_action_counter += 1

            reward[rew_index] += (components["sprint_bonus"][rew_index] + components["stop_sprint_bonus"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
