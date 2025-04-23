import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for offensive skills including passing, shooting, 
    and dribbling in order to develop advanced offensive strategies.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.precision_reward = 0.5  # Reward for precise shooting
        self.dribbling_reward = 0.3  # Reward for effective dribbling
        self.passing_reward = 0.2    # Reward for successful passing
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and all counters.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save wrapper state to pickle
        """
        to_pickle['CheckpointRewardWrapper'] = dict(
            precision_reward=self.precision_reward,
            dribbling_reward=self.dribbling_reward,
            passing_reward=self.passing_reward
        )
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore wrapper state from pickle
        """
        from_pickle = self.env.set_state(state)
        self.precision_reward = from_pickle['CheckpointRewardWrapper']['precision_reward']
        self.dribbling_reward = from_pickle['CheckpointRewardWrapper']['dribbling_reward']
        self.passing_reward = from_pickle['CheckpointRewardWrapper']['passing_reward']
        return from_pickle

    def reward(self, reward):
        """
        Customize reward based on offensive actions performed.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "precision_reward": [0.0] * len(reward),
            "dribbling_reward": [0.0] * len(reward),
            "passing_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Precision shooting: simple heuristic goal proximity detection
            if (o['ball'][0] > 0.9 and abs(o['ball'][1]) <= 0.08):
                components["precision_reward"][rew_index] = self.precision_reward
                reward[rew_index] += self.precision_reward

            # Dribbling reward
            if ('sticky_actions' in o and o['sticky_actions'][9] == 1 and o['ball_owned_team'] == 0):
                components["dribbling_reward"][rew_index] = self.dribbling_reward
                reward[rew_index] += self.dribbling_reward

            # Passing effectiveness - looking to reward long passes moving the ball significantly upfield
            if ('ball_direction' in o and o['ball_direction'][0] > 0.1 and abs(o['ball_direction'][1]) > 0.1):
                components["passing_reward"][rew_index] = self.passing_reward
                reward[rew_index] += self.passing_reward 

        return reward, components

    def step(self, action):
        """
        Execute action, modify reward based on the action and return observation, 
        modified reward, flag if the game is done, and any other information to the user.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        for agent_obs in observation:
            for i, action_status in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_status
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
