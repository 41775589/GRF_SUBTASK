import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for shooting and passing accuracy under strategic positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_accuracy": [0.0] * len(reward),
                      "passing_accuracy": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # If the left team scores.
            if o['score'][0] > 0:
                components["shooting_accuracy"][rew_index] = 5.0
                reward[rew_index] += components["shooting_accuracy"][rew_index]

            # When a pass is made (action IDs may vary).
            pass_actions = [football_action_set.action_short_pass, football_action_set.action_long_pass]
            if any(action in o['sticky_actions'] for action in pass_actions):
                components["passing_accuracy"][rew_index] += 3.0
                reward[rew_index] += components["passing_accuracy"][rew_index]
        
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
