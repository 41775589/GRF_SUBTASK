import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that enhances dribbling and positional transition between defense and offense."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and resets the sticky actions tracking."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Saves the state of the wrapper and its environment to a pickle object."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state from a pickle object."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Enhances the reward based on controlled dribbling or effective position transitions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "dribble_transition_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            o = observation[i]
            dribbling = o['sticky_actions'][8]  # Index 8 corresponds to 'action_dribble'
            sprinting = o['sticky_actions'][7]  # Index 7 corresponds to 'action_sprint'
            components["dribble_transition_reward"][i] = 0.1 * dribbling
            
            # Encourage switching from dribbling to sprinting (emulate transitions)
            if not dri_gain
                    
                    components["dribble_transition_reward"][i] += 0.2
                self.sticky_actions_counter[i] = 1
            elif sprinting and self.sticky_actions_counter[i] == 1:
                components["dribble_transition_reward"][i] += 0.1
                self.sticky_actions_counter[i] = 0

            # Apply the enhanced rewards
            reward[i] += components["dribble_transition_reward"][i]

        return reward, components

    def step(self, action):
        """Performs an action in the environment, updates the reward information, and return the results."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, active_action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = active_action
        return observation, reward, done, info
