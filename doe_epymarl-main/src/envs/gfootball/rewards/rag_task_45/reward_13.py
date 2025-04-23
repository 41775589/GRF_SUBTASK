import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for performing stop movements 
    and quickly changing directions which is essential in defensive maneuvers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_movement_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Assume the environment reuses a list for sticky_actions in observations
            sticky_action_changes = o['sticky_actions'] - self.sticky_actions_counter
            self.sticky_actions_counter = o['sticky_actions'].copy()

            # Reward for stopping movement
            if sticky_action_changes[0] == 1 or sticky_action_changes[4] == 1:
                components["stop_movement_reward"][rew_index] = 0.2
                reward[rew_index] += components["stop_movement_reward"][rew_index]

            # Reward for quick direction changes (left-right)
            if sticky_action_changes[0] == 1 and sticky_action_changes[4] == 1:
                reward[rew_index] += 0.3
                components["stop_movement_reward"][rew_index] += 0.3

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        info["reward_components"] = {
            key: sum(value) for key, value in components.items()
        }
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
