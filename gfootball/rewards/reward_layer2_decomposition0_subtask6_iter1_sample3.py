import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward based on defensive positioning and sliding tackle execution."""

    def __init__(self, env):
        super().__init__(env)
        self.sliding_tackle_reward = 0.5
        self.defensive_position_reward = 0.2
        self.sliding_tackle_count = 0

    def reset(self):
        self.sliding_tackle_count = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sliding_tackle_count'] = self.sliding_tackle_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sliding_tackle_count = from_pickle['sliding_tackle_count']
        return from_pickle

    def reward(self, reward):
        """Adjust the reward based on the agent's defensive actions, particularly sliding tackles."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "sliding_tackle_reward": [0.0], "defensive_position_reward": [0.0]}

        if observation is None:
            return reward, components
        
        o = observation[0]
        components["base_score_reward"][0] = reward
        
        # Check if the agent performs a sliding tackle
        if 'sticky_actions' in o and o['sticky_actions'][7] == 1:  # Assuming index 7 is 'action_sliding'
            components["sliding_tackle_reward"][0] = self.sliding_tackle_reward
            reward += components["sliding_tackle_reward"][0]
            self.sliding_tackle_count += 1

        # Reward defensive positioning based on distance to own goal
        if 'left_team' in o:
            my_position = o['left_team'][o['active']]
            goal_position = [-1, 0]  # Assuming the goal is at the left end
            distance_to_goal = np.linalg.norm(np.array(my_position) - np.array(goal_position))
            if distance_to_goal < 0.5:  # closer to the goal is better in terms of positioning
                components["defensive_position_reward"][0] = self.defensive_position_reward * (0.5 - distance_to_goal)
                reward += components["defensive_position_reward"][0]

        return [reward], components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward[0]

        # Add the components to info for analysis
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward[0], done, info
