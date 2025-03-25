import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on the precision and control required for executing high trajectory passes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_threshold = 0.02  # Arbitrary threshold for "high precision"
        self.max_distance_reward = 1.0
        self.max_height_reward = 1.0

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
                      "precision_reward": [0.0] * len(reward),
                      "height_control_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Evaluate precision of the pass
            if ('ball_direction' in o and 'ball_owned_team' in o 
                and o['ball_owned_team'] == 0):  # Assuming '0' is the team index of agent
                ball_vel = np.linalg.norm(o['ball_direction'][0:2])  # ignore z component for velocity
                if ball_vel < self.pass_accuracy_threshold:
                    components['precision_reward'][rew_index] = self.max_distance_reward * (self.pass_accuracy_threshold - ball_vel) / self.pass_accuracy_threshold

            # Evaluate height control
            if 'ball' in o:
                ball_height = o['ball'][2]
                ideal_height = 0.5  # Assume 0.5 is the ideal height for a pass
                height_difference = abs(ball_height - ideal_height)
                if height_difference < self.pass_accuracy_threshold:
                    components['height_control_reward'][rew_index] = self.max_height_reward * (1 - height_difference / self.pass_accuracy_threshold)

            # Combine rewards
            reward[rew_index] += components['precision_reward'][rew_index] + components['height_control_reward'][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
