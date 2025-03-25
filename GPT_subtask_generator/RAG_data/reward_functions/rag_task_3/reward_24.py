import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a shooting practice reward, focusing on accuracy and power."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.base_reward_for_shot_accuracy = 0.5
        self.base_reward_for_shot_power = 0.5

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
                      "shot_accuracy_reward": [0.0] * len(reward),
                      "shot_power_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_direction' not in o or 'ball_owned_player' not in o:
                continue
            # Check if the active player has made a shot
            if o['sticky_actions'][9]:  # Assuming index 9 corresponds to the 'shot' action
                ball_speed = np.linalg.norm(o['ball_direction'])
                shot_accuracy = 1 - min(1, abs(o['ball'][1]))  # Assuming goal y-range is -1 to 1, center 0

                components["shot_accuracy_reward"][rew_index] = self.base_reward_for_shot_accuracy * shot_accuracy
                components["shot_power_reward"][rew_index] = self.base_reward_for_shot_power * min(1, ball_speed)

                reward[rew_index] += components["shot_accuracy_reward"][rew_index]
                reward[rew_index] += components["shot_power_reward"][rew_index]

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
