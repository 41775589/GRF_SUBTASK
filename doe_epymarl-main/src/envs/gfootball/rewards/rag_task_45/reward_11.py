import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper class that adds rewards based on the 'Stop-Sprint' and 'Stop-Moving' actions,
    promoting abrupt stopping abilities for quick defensive maneuvers."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To count the number of sticky actions

    def reset(self):
        """Reset the environment and clear the action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Include wrapped state components when saving the environment's state."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state, including any state managed by the wrapper."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Augment the reward based on defensive stopping actions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(), 
            "stop_sprint_reward": [0.0] * len(reward),
            "stop_moving_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for index, o in enumerate(observation):
            # Check if the previous action was sprint and now it's stopped
            if o['sticky_actions'][8] == 0 and self.sticky_actions_counter[8] > 0:
                components["stop_sprint_reward"][index] = 0.1
                reward[index] += components["stop_sprint_reward"][index]
            
            # Check if the player has stopped moving (all movement actions are 0)
            if np.sum(o['sticky_actions'][:8]) == 0 and np.sum(self.sticky_actions_counter[:8]) > 0:
                components["stop_moving_reward"][index] = 0.1
                reward[index] += components["stop_moving_reward"][index]

            # Update the sticky action counters for tracking changes
            self.sticky_actions_counter = o['sticky_actions']

        return reward, components

    def step(self, action):
        """Perform a step, augment the returned info with reward components, and reset sticky actions."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
