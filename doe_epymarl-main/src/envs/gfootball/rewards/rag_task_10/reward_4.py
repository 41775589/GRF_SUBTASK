import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized checkpoint reward focusing on defensive actions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_action_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Defensive actions are indexed according to known sticky actions:
            # Assume indexes 7, 8, 9 correspond to Slide, Stop-Dribble, Stop-Moving
            defensive_actions = o['sticky_actions'][7:10]
            if any(defensive_actions):
                defensive_action_bonus = 0.05  # Reward increment for defensive actions
                # Count number of defensive actions
                defense_count = np.sum(defensive_actions)
                
                # Update components with defensive reward
                components["defensive_action_reward"][rew_index] = defensive_action_bonus * defense_count
                # Add defensive action bonus to reward
                reward[rew_index] += components["defensive_action_reward"][rew_index]        
            
        return reward, components

    def step(self, action):
        # Obtain the result from the environment's step function
        observation, reward, done, info = self.env.step(action)

        # Modify the reward using the custom reward method defined
        reward, components = self.reward(reward)

        # Sum the rewards and add additional info about the reward components
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions counter
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        sticky_array_indices = [7, 8, 9]  # Assuming these indices correspond to the targeted defensive actions

        for agent_obs in obs:
            for i in sticky_array_indices:
                action = agent_obs['sticky_actions'][i]
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
