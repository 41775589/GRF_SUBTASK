import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to enhance the learning of sliding tackles for a defender agent,
    specifically enhancing detection of successful tackles and team defensive positioning.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_success_count = 0
        self.last_ball_owner = None
        self.progress_toward_opponent_goal = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sliding_success_count = 0
        self.last_ball_owner = None
        self.progress_toward_opponent_goal = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sliding_success_count': self.sliding_success_count,
            'last_ball_owner': self.last_ball_owner,
            'progress_toward_opponent_goal': self.progress_toward_opponent_goal
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sliding_success_count = from_pickle['CheckpointRewardWrapper']['sliding_success_count']
        self.last_ball_owner = from_pickle['CheckpointRewardWrapper']['last_ball_owner']
        self.progress_toward_opponent_goal = from_pickle['CheckpointRewardWrapper']['progress_toward_opponent_goal']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()[0]  # Assuming a single agent
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components

        # Monitor sliding action and transitions in ball control
        current_ball_owner = observation['ball_owned_player']
        if self.last_ball_owner != current_ball_owner and current_ball_owner == observation['active']:
            if observation['sticky_actions'][9]:  # index 9 is for sliding
                self.sliding_success_count += 1
                reward[0] += 0.5  # additional reward for successful sliding

        # Reward progress toward the opponent's goal
        if observation['ball'][0] > self.progress_toward_opponent_goal:
            reward[0] += 0.1 * (observation['ball'][0] - self.progress_toward_opponent_goal)  # reward forward movement
            self.progress_toward_opponent_goal = observation['ball'][0]

        self.last_ball_owner = current_ball_owner
        components["sliding_success"] = [self.sliding_success_count * 0.5]
        components["progress_toward_goal"] = [self.progress_toward_opponent_goal]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
