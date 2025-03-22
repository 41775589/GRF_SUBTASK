import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Reward when goal scored
        self.goal_score_reward = 1.0
        # A reward for making a shot on goal
        self.shot_on_goal_reward = 0.3
        # A reward for successful dribbles approaching the opponent goal
        self.dribble_approach_reward = 0.1
        # Reward component for successful long passes
        self.long_pass_reward = 0.2
        # Initialize rewards achieved per step
        self._cumulative_rewards = []

    def reset(self):
        """Reset the rewards achieved list."""
        self._cumulative_rewards = []
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the state of cumulative rewards."""
        to_pickle['CumulativeRewards'] = self._cumulative_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from pickle and retrieve cumulative rewards state."""
        from_pickle = self.env.set_state(state)
        self._cumulative_rewards = from_pickle.get('CumulativeRewards', [])
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on the observed actions and their outcomes."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goal_score_reward": 0.0,
            "shot_on_goal_reward": 0.0,
            "dribble_approach_reward": 0.0,
            "long_pass_reward": 0.0
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Assuming game modes definitions:
            # FREE_KICK = 3, NORMAL_PLAY = 0,
            # Here we assume that a FREE_KICK leads to either a long pass or goal directly
            
            # Check if a goal was scored
            if o['score'][0] > 0:  # Assuming score for the team controlled by the agents is index 0
                reward[rew_index] += self.goal_score_reward
                components["goal_score_reward"] = self.goal_score_reward

            # Check for shots towards the goal
            if o['game_mode'] == 3 and np.linalg.norm(o['ball'] - np.array([1, 0])) < 0.1:
                reward[rew_index] += self.shot_on_goal_reward
                components["shot_on_goal_reward"] = self.shot_on_goal_reward
            
            # Assume there is a method to calculate dribbling towards opponent's goal
            if 'dribble_effective' in o:
                reward[rew_index] += self.dribble_approach_reward
                components["dribble_approach_reward"] = self.dribble_approach_reward
            
            # Assume definition of a long successful pass
            if 'successful_long_pass' in o:
                reward[rew_index] += self.long_pass_reward
                components["long_pass_reward"] = self.long_pass_reward

        return reward, components

    def step(self, action):
        # Call the original step method
        observation, reward, done, info = self.env.step(action)
        # Modify the reward using the reward() method
        reward, components = self.reward(reward)
        # Add final reward to the info
        info["final_reward"] = sum(reward)

        # Traverse the components dictionary and write each key-value to info['component_']
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
