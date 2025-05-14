import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to enhance the learning of sliding tackles for a defender agent."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_success_count = 0
        self.last_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_success_count = 0
        self.last_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sliding_success_count': self.sliding_success_count,
            'last_ball_owner': self.last_ball_owner
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sliding_success_count = from_pickle['CheckpointRewardWrapper']['sliding_success_count']
        self.last_ball_owner = from_pickle['CheckpointRewardWrapper']['last_ball_owner']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None or 'left_team' not in observation:
            return reward, components

        # Check for sliding action success
        current_ball_owner = observation['ball_owned_player']
        if self.last_ball_owner is not None and self.last_ball_owner != current_ball_owner:
            if observation['sticky_actions'][9] == 1:  # index 9 is for sliding
                self.sliding_success_count += 1
                reward[0] += 0.5  # additional reward for successful sliding
        
        self.last_ball_owner = current_ball_owner
        components["sliding_success"] = [self.sliding_success_count * 0.5]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
