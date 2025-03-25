import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on offensive skills enhancement."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_efficiency = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = None  # No state-specific data to restore in this example
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # State restoration implementation if needed.
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_efficiency_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['score'][0] > o['score'][1]:  # Assuming the agent controls the left team
                reward[rew_index] += 1  # Reward for scoring a goal

            # Check for possession and passing efficiency
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                reward[rew_index] += self.pass_efficiency

                # Assume high and long passes trigger a specific action number, e.g., 7 or 8
                # We evaluate sticky actions to check for the type of pass
                if o['sticky_actions'][6] or o['sticky_actions'][7]:
                    reward[rew_index] += self.pass_efficiency  # Reward for high and long passes
                    components["pass_efficiency_reward"][rew_index] += self.pass_efficiency

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
                self.sticky_actions_counter[i] += int(action)
        return observation, reward, done, info
