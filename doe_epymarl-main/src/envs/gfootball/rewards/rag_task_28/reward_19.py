import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward function to focus on dribbling skills, especially in one-on-one situations with the goalkeeper."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_reward = 0.2
        self.direction_change_reward = 0.1
        self.close_to_goalkeeper_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "ball_control_reward": [0.0] * len(reward),
                      "direction_change_reward": [0.0] * len(reward),
                      "close_to_goalkeeper_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage ball possession
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                reward[rew_index] += self.ball_control_reward
                components["ball_control_reward"][rew_index] = self.ball_control_reward

            # Reward change of direction with the ball
            if o['ball_owned_team'] == 0 and o['sticky_actions'][6] != self.sticky_actions_counter[6]:
                reward[rew_index] += self.direction_change_reward
                components["direction_change_reward"][rew_index] = self.direction_change_reward
                self.sticky_actions_counter[6] = o['sticky_actions'][6]

            # Additional reward for being close to the goalkeeper in a goal-scoring position
            if np.linalg.norm(o['ball'] - np.array([1, 0])) < 0.2:
                reward[rew_index] += self.close_to_goalkeeper_reward
                components["close_to_goalkeeper_reward"][rew_index] = self.close_to_goalkeeper_reward
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
