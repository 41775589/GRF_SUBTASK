import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for effective long-range passing and strategic play."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "long_pass_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        # Parameters for rewarding criteria
        long_pass_threshold = 0.3  # Threshold distance for a "long range" pass
        pass_success_bonus = 0.1   # Bonus reward for a successful long range pass

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0:  # Check if the left team owns the ball
                if o['ball_direction'][0] > long_pass_threshold or o['ball_direction'][1] > long_pass_threshold:
                    # Reward a pass if the change in position is greater than the threshold
                    components["long_pass_bonus"][rew_index] = pass_success_bonus

        # Include the component bonuses into the final reward
        reward = [base + bonus for base, bonus in zip(components["base_score_reward"], components["long_pass_bonus"])]
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
