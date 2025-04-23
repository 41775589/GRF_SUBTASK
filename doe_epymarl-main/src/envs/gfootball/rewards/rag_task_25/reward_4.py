import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on dribbling and sprinting capabilities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            dribble_action_index = 9  # Assuming 9 is the index for dribble action
            sprint_action_index = 8   # Assuming 8 is the index for sprint action

            # Encourage dribbling when closely guarded
            if o['sticky_actions'][dribble_action_index] == 1 and o['ball_owned_team'] == 0:
                components["dribble_reward"][rew_index] += 0.05

            # Encourage effective use of sprints
            if o['sticky_actions'][sprint_action_index] == 1:
                nearby_opponents = np.linalg.norm(o['right_team'] - o['ball'], axis=1) < 0.1
                if any(nearby_opponents):
                    components["sprint_reward"][rew_index] += 0.1

            # Combine the components
            reward[rew_index] += components["dribble_reward"][rew_index] + components["sprint_reward"][rew_index]

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
