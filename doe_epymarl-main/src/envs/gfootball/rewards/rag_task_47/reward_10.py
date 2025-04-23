import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that emphasizes mastering sliding tackles during defensive counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.sliding_tackle_counter = 0
        self.sliding_tackle_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.sliding_tackle_counter = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sliding_tackle_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_position = o['ball']
            if self.last_ball_position is not None:
                ball_travelled_distance = np.linalg.norm(
                    np.array(current_ball_position[:2]) - np.array(self.last_ball_position[:2]))

                if 'sticky_actions' in o:
                    if o['sticky_actions'][9] == 1: # 9 is the index for 'action_sliding_tackle'.
                        components["sliding_tackle_reward"][rew_index] = self.sliding_tackle_reward
                        reward[rew_index] += components["sliding_tackle_reward"][rew_index]

                # Encourage more sliding tackles if ball is closer and in our defensive third.
                if (current_ball_position[0] < 0.4) and ball_travelled_distance > 0.1:
                    self.sliding_tackle_counter += 1
                    components["sliding_tackle_reward"][rew_index] = self.sliding_tackle_reward

            reward[rew_index] += components["sliding_tackle_reward"][rew_index]
            self.last_ball_position = current_ball_position

        return reward, components

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "last_ball_position": self.last_ball_position,
            "sliding_tackle_counter": self.sliding_tackle_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']['last_ball_position']
        self.sliding_tackle_counter = from_pickle['CheckpointRewardWrapper']['sliding_tackle_counter']
        return from_pickle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
