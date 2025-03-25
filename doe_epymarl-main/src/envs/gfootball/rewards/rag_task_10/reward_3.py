import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward function to focus on defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_acts_reward = 0.2  # Additional reward for defensive actions

    def reset(self):
        """ Reset the sticky actions counter on environment reset. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Store environment state additions here. """
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set environment state from the saved state. """
        return self.env.set_state(state)

    def reward(self, reward):
        """ Reward augmentation focusing on defensive actions. """
        
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            old_reward = reward[rew_index]
            components["defensive_action_reward"] = [0.0] * len(reward)
            
            # Check for defensive actions via sticky actions
            if 'sticky_actions' in o:
                defensive_actions = {'action_sliding_tackle': 0, 'action_right': 4, 'action_left': 0}
                actions_count = np.array(o['sticky_actions'])
                
                # Apply reward for defensive actions based on their indexes in the sticky actions
                for idx, count in defensive_actions.items():
                    curr = actions_count[count]
                    past = self.sticky_actions_counter[count]
                    if curr > past:
                        reward[rew_index] += self.defensive_acts_reward
                        components["defensive_action_reward"][rew_index] += self.defensive_acts_reward
                        self.sticky_actions_counter[count] = curr

            # Print debugging info
            print("Old Reward:", old_reward, " New Reward:", reward[rew_index])

        return reward, components

    def step(self, action):
        """ Capturing output from the environment's step method and appends reward adjustments. """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
