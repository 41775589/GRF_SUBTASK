import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that applies a reward function focusing on precise Stop-Dribble tactics under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Reset the sticky actions counter along with the environment """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Get the state of the environment for serialization """
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set the state of the environment for deserialization """
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """ Reward function for Stop-Dribble strategy application under pressure """
        observation = self.env.unwrapped.observation()
        components = {
          "base_score_reward": reward.copy(),
          "stop_dribble_under_pressure_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'sticky_actions' in o:
                dribble_action = o['sticky_actions'][9]
                # Apply additional reward for stopping dribble under pressure
                if dribble_action == 1:  # Checking if dribble action is active
                    components["stop_dribble_under_pressure_reward"][rew_index] = 0.1
                    reward[rew_index] += components["stop_dribble_under_pressure_reward"][rew_index]

        return reward, components

    def step(self, action):
        """ Process environment step including action execution and observation """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        return observation, reward, done, info
