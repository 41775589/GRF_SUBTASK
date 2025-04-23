import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards the use of dribbling and sprints effectively."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self._dribble_reward = 0.05
        self._sprint_dribble_combination_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._collected_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._collected_rewards = from_pickle.get('CheckpointRewardWrapper', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_dribble_combination_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            if 'sticky_actions' in obs:
                sticky_actions = obs['sticky_actions']
                if sticky_actions[9] == 1:  # dribble action
                    components["dribble_reward"][rew_index] = self._dribble_reward
                    reward[rew_index] += components["dribble_reward"][rew_index]

                if sticky_actions[8] == 1 and sticky_actions[9] == 1:  # sprint and dribble together
                    components["sprint_dribble_combination_reward"][rew_index] = self._sprint_dribble_combination_reward
                    reward[rew_index] += components["sprint_dribble_combination_reward"][rew_index]

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
        return observation, reward, done, info
