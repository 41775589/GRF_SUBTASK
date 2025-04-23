import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic midfield control reward based on player roles and positions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Processing observations
        for rew_index, o in enumerate(observation):
            # Evaluate role contribution in offense and defense transitions
            midfield_contrib = 0
            
            # Checking for central midfield roles
            central_roles = [5]  # Assuming role index '5' corresponds to central midfield 
            for role_index in central_roles:
                if role_index in o['left_team_roles'] or role_index in o['right_team_roles']:
                    midfield_contrib += 0.1  # Reward contribution from central midfield
            
            # Checking for wide midfield roles
            wide_roles = [6, 7]  # Assuming roles '6' and '7' correspond to left and right midfield respectively
            for role_index in wide_roles:
                if role_index in o['left_team_roles'] or role_index in o['right_team_roles']:
                    midfield_contrib += 0.05  # Reward contribution for side field control
            
            # Adjusting reward based on midfield control
            reward[rew_index] += midfield_contrib
            components["midfield_control_reward"][rew_index] = midfield_contrib

        return reward, components

    def step(self, action):
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
