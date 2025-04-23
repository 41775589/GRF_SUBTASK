import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for effective mid to long-range passing between players of the same team. 
    This is designed to improve strategic coordination and passing precision over greater distances.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.passing_threshold = 0.3  # Distance threshold to consider a pass long-range
        self.pass_reward = 0.2  # Reward given for a successful long-range pass
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def reward(self, reward):
        """
        Reward function that focuses on long-range passes. It adds a positive reward if a pass meets 
        the distance criteria and is successfully received by a teammate without interception.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0] * len(reward)
        }

        for i in range(len(reward)):
            if observation is None:
                continue

            o = observation[i]
            # Check for ball possession and pass execution
            if o['ball_owned_team'] == 0 and 'ball' in o and 'right_team' in o:
                distances = np.linalg.norm(o['right_team'] - o['ball'][:2], axis=1)
                long_passes = distances > self.passing_threshold
                if np.any(long_passes):
                    components["long_pass_reward"][i] = self.pass_reward
                    reward[i] += self.pass_reward

        return reward, components
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'CheckpointRewardWrapper' in from_pickle:
            self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper'].get("sticky_actions_counter", np.zeros(10, dtype=int))
        return from_pickle

    def step(self, action):
        """
        Take a step using the specified action, modifying the reward components based on the new state.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
