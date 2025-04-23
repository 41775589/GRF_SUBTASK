import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful dribbling and dynamic positioning transitions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_start_reward = 0.05
        self.position_transition_reward = 0.03

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_start_reward": [0.0] * len(reward),
                      "position_transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Check for starting dribble
            if o['sticky_actions'][9] == 1 and self.sticky_actions_counter[9] == 0:
                components["dribble_start_reward"][rew_index] = self.dribble_start_reward
                reward[rew_index] += components["dribble_start_reward"][rew_index]

            # Transition between defensive to offensive positioning
            if o['ball_owned_team'] == 1 and o['right_team'][o['active']][0] > 0.5:
                components["position_transition_reward"][rew_index] = self.position_transition_reward
                reward[rew_index] += components["position_transition_reward"][rew_index]

            # Update sticky actions counter
            self.sticky_actions_counter = o['sticky_actions']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"reward_component_{key}"] = sum(value)
        return observation, reward, done, info
