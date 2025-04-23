import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies base reward to enhance focus on dribbling and
    dynamic positioning for transitions between defense and offense.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_count = np.zeros((2,), dtype=int)  # Track dribbling duration

    def reset(self):
        """
        Reset the environment state and dribbling count.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_count.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the additional dribble count state.
        """
        to_pickle['dribble_count'] = self.dribble_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the dribble count state.
        """
        from_pickle = self.env.set_state(state)
        self.dribble_count = from_pickle['dribble_count']
        return from_pickle

    def reward(self, reward):
        """
        Custom reward function that emphasizes dribbling and dynamic positioning.
        """
        observation = self.env.unwrapped.observation()
        new_rewards = reward.copy()
        components = {"base_score_reward": reward.copy(), 
                      "dribbling_reward": [0.0, 0.0], 
                      "positioning_reward": [0.0, 0.0]}

        for i, o in enumerate(observation):
            # Reward for dribbling (action index 9 is dribble)
            if o['sticky_actions'][9]:  # dribbling
                if self.dribble_count[i] < 10:  # adding limiting condition to prevent infinite dribble reward
                    self.dribble_count[i] += 1
                    components["dribbling_reward"][i] = 0.02 * self.dribble_count[i]
            
            # Transitional dynamic positioning reward based on position change
            if 'left_team_direction' in o and i == o['active']:
                movement_magnitude = np.linalg.norm(o['left_team_direction'][i])
                components["positioning_reward"][i] = movement_magnitude * 0.005  # Small reward for movement

            # Update reward
            new_rewards[i] += components["dribbling_reward"][i] + components["positioning_reward"][i]

        return new_rewards, components

    def step(self, action):
        """
        Executes a step in the environment and applies reward modification.
        """
        obs, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info["final_reward"] = sum(new_reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        self.sticky_actions_counter.fill(0)
        current_obs = self.env.unwrapped.observation()
        for agent_obs in current_obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active

        return obs, new_reward, done, info
