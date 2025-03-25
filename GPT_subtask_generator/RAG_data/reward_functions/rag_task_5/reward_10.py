import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on refining defensive skills and improving transition to counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        """ Reset the environment and sticky actions counter. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Save current wrapper state to pickle. """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Load wrapper state from pickle. """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions'])
        return from_pickle

    def reward(self, reward):
        """ Customize reward function to train defensive behaviors and counter-attacks. """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for i, o in enumerate(observation):
            # Reward increased when the ball is reclaimed from the opponent in their half
            if o['ball_owned_team'] == 1 and o['ball'][0] < 0:  # Ball in opponent's half and owned by right team
                components["defensive_reward"][i] += 0.3
                
            # Reward quick transition: gaining roles after stealing the ball
            if o['ball_owned_team'] == 0 and self.sticky_actions_counter[8] > 0:  # Sprint action is active
                components["counter_attack_reward"][i] += 0.5
            
            # Compute final reward
            reward[i] = (components["base_score_reward"][i] +
                         components["defensive_reward"][i] +
                         components["counter_attack_reward"][i])

        return reward, components

    def step(self, action):
        """ 
        Take an environment step and wrap the reward function, tracking sticky actions and providing detailed debug info. 
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = action_active
        return observation, reward, done, info
