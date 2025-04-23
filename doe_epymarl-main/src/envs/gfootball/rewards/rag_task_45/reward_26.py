import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for practicing Stop-Sprint and Stop-Moving techniques."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # Context from the environment's observation
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defense_skill_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            sticky_actions = o['sticky_actions']
            action_stop = sticky_actions[0]  # assuming 0 is the index for stop action
            action_sprint = sticky_actions[8]  # assuming 8 is the index for sprint action

            # Encouraging sudden stopping following a sprint
            if action_stop and self.sticky_actions_counter[8] > 0:
                components["defense_skill_reward"][rew_index] += 0.2
                reward[rew_index] += 0.2 * components["defense_skill_reward"][rew_index]

            # Update the sticky actions counter: especially track sprint actions
            self.sticky_actions_counter += sticky_actions
            # Reset sprint counter if stop action is detected
            if action_stop:
                self.sticky_actions_counter[8] = 0

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
