import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds custom rewards focusing on abrupt stopping and sprinting effectively for defensive maneuvers."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._stop_action_reward = 0.1
        self._sprint_action_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_action_reward": [0.0] * len(reward),
                      "sprint_action_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward for utilizing the stop action skillfully
            if o['sticky_actions'][0] == 1:  # Assuming index 0 is stop action
                components["stop_action_reward"][rew_index] = self._stop_action_reward
                reward[rew_index] += components["stop_action_reward"][rew_index]

            # Reward for effective use of sprint in defensive maneuvers
            if o['sticky_actions'][8] == 1:  # Assuming index 8 is sprint action
                components["sprint_action_reward"][rew_index] = self._sprint_action_reward
                reward[rew_index] += components["sprint_action_reward"][rew_index]

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
