import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a midfield dynamics task-specific reward."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_dynamics_coeff = {0: 0.1, 1: 0.15}

    def reset(self):
        """Reset the environment and clear the midfield counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state and include midfield dynamics data."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the current state and midfield dynamics data from state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Calculate reward with additional midfield dynamics reward."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.array(reward).copy(),
                      "midfield_dynamics_reward": np.zeros(len(reward))}
                      
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Reward proportional to the player's y-position in midfield
            mid_y_pos = abs(o['left_team'][o['active']][1])
            if mid_y_pos < 0.2 and o['left_team_direction'][o['active']][0] > 0:
                # Reward for moving forward in the midfield under pressure
                components["midfield_dynamics_reward"][i] += self.midfield_dynamics_coeff[0]

            # Include a positional awareness bonus for switching fields smoothly
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0 and
                o['ball'][1] * o['ball_direction'][1] < 0):
                components["midfield_dynamics_reward"][i] += self.midfield_dynamics_coeff[1] * abs(o['ball_direction'][1])

            reward[i] += components["midfield_dynamics_reward"][i]

        return reward, components

    def step(self, action):
        """Step the environment and enhance the reward with midfield dynamics bonus."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = np.sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = np.sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
