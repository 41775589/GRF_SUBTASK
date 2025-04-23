import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward focused on rewarding advanced ball control and passing under pressure.
    It specifically rewards Short Pass, High Pass, and Long Pass during tight game situations.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_threshold = 0.8  # Threshold for considering a pass successful
        self.pressure_threshold = 1.0       # Distance threshold for considering the player under pressure
        self.pass_reward = 0.5              # Reward for successful passes under pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
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

        assert len(reward) == len(observation)

        for rew_index, r in enumerate(reward):
            o = observation[rew_index]
            nearby_opponents = np.sum(np.sqrt(np.sum((o['left_team'][o['designated']] - o['right_team']) ** 2, axis=1)) < self.pressure_threshold)

            if ('sticky_actions' in o and nearby_opponents > 0 and r > 0.0):
                # Assuming successful pass actions indices correspond to 1, 2, 3
                if o['sticky_actions'][1] or o['sticky_actions'][2] or o['sticky_actions'][3]:
                    # Check if the pass is accurate and under pressure
                    ball_travel_distance = np.linalg.norm(o['ball_direction'][:2])
                    if ball_travel_distance >= self.pass_accuracy_threshold:
                        components['passing_reward'][rew_index] = self.pass_reward
                        reward[rew_index] += components['passing_reward'][rew_index]

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
