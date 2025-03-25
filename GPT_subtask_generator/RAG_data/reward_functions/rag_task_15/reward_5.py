import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for technical aspects and accuracy of long passes.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.2

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
                      "long_pass_accuracy": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] in {0, 2, 3, 4, 5, 6} and o['ball_owned_team'] == 0:
                dist = np.linalg.norm(o['ball_direction'][:2]) # Considering only x and y
                # Reward calculated based on distance covering, assuming long passes > 0.5 in the normalized field
                if dist > 0.5:
                    components["long_pass_accuracy"][rew_index] = self._checkpoint_reward
                    reward[rew_index] += components["long_pass_accuracy"][rew_index]
        
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
