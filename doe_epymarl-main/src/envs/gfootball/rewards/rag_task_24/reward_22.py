import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the training of agents in mid to long-range passing with precision
    and strategic use of such passes in coordinated plays.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_quality_threshold = 0.7  # 70% of the field length is considered long-range
        self.pass_precision_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for index, o in enumerate(observation):
            if 'ball_direction' not in o or 'ball_owned_team' not in o:
                continue

            ball_speed = np.linalg.norm(o['ball_direction'][:2])  # Only consider X,Y components
            if o['ball_owned_team'] == 1 and ball_speed > self.pass_quality_threshold:
                # Reward passes that exceed the speed threshold i.e., long-range and well-directed passes.
                components['passing_reward'][index] += self.pass_precision_reward

            reward[index] += components['passing_reward'][index]

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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
