import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that emphasizes dribbling and use of sprints in play.
    This is aimed at agents that attempt to refine close ball control and rapid 
    dynamic movements, particularly important for offensive actions.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # for tracking sticky actions
        self._dribble_reward_coefficient = 0.05
        self._sprint_reward_coefficient = 0.1

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the state of the sticky action counters with the environment state.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state from the pickle object, including the sticky actions counter.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modify the reward based on dribbling and sprinting actions usage.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}

        if not observation:
            return reward, components

        for rew_index, (reward_val, obs) in enumerate(zip(reward, observation)):
            active = obs.get('active', -1)
            if active == -1:  # Check if there is a player actively controlled
                continue
            
            sticky_actions = obs.get('sticky_actions', [])
            
            # Reward for dribbling: Active dribbling action increases reward by specified coefficient
            if sticky_actions[9]:  # index 9 corresponds to dribble action
                components["dribble_reward"][rew_index] = self._dribble_reward_coefficient
                reward_val += components["dribble_reward"][rew_index]

            # Reward for sprinting: Active sprint action increases reward by specified coefficient
            if sticky_actions[8]:  # index 8 corresponds to sprint action
                components["sprint_reward"][rew_index] = self._sprint_reward_coefficient
                reward_val += components["sprint_reward"][rew_index]
            
            reward[rew_index] = reward_val

        return reward, components

    def step(self, action):
        """
        Execute a step using the given action, modify the reward, and track sticky actions.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
