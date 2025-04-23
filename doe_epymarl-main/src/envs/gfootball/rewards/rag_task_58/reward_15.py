import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward function based on defensive and passing performance.
    We're particularly interested in:
    - Encouraging keeping ball possession under pressure.
    - Rewarding successful pass completion.
    - Rewarding interception of opponent's passes.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = [0, 0]  # Placeholder to track interceptions for each team.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.interceptions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.interceptions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_reward": [0.0] * len(reward),
                      "pass_completion_reward": [0.0] * len(reward),
                      "interception_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for maintaining possession under pressure
            if o['ball_owned_team'] == o['active']:
                components["possession_reward"][rew_index] += 0.01

            # Reward for successful pass completions
            if 'action' in o and o['action'] == "pass" and o['ball_owned_team'] == o['active']:
                components["pass_completion_reward"][rew_index] += 0.2
            
            # Check for interceptions and reward them
            if o['ball_owned_team'] == 1 - o['active']:  # Opponent has the ball
                if 'previous_action' in o and o['previous_action'] == "pass":  # Latest was a pass
                    components["interception_reward"][rew_index] += 0.3
                    self.interceptions[o['active']] += 1

            reward[rew_index] += (components["possession_reward"][rew_index] +
                                  components["pass_completion_reward"][rew_index] +
                                  components["interception_reward"][rew_index])

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
