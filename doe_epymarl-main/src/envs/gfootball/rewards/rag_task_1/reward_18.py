import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on offensive maneuvers during varied game phases."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for reward calculation
        self.offensive_positions = {
            'attacking_third': 0.67,  # threshold for being considered in the attacking third
            'goal_range': 0.95        # near opponent's goal
        }
        self.rewards = {
            'attacking': 0.2, 
            'goal_proximity': 0.4, 
            'goal_scored': 1.0
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "attacking_reward": [0.0] * len(reward),
                      "goal_proximity_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage forwarding the ball into the attacking third
            if o['ball'][0] >= self.offensive_positions['attacking_third']:
                components["attacking_reward"][rew_index] = self.rewards['attacking']
            
            # Extra reward for being in close proximity to the opponent's goal
            if o['ball'][0] >= self.offensive_positions['goal_range']:
                components["goal_proximity_reward"][rew_index] = self.rewards['goal_proximity']

            # Calculate overall reward
            total_reward = (reward[rew_index] +
                            components["attacking_reward"][rew_index] +
                            components["goal_proximity_reward"][rew_index])
            # Ensuring goals have a significant reward
            if o['score'][0] > o['score'][1]:  # If left team (assuming control) scores
                total_reward += self.rewards['goal_scored']

            reward[rew_index] = total_reward

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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
