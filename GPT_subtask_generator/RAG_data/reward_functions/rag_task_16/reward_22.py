import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a precision pass reward focused on executing high passes."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_score_reward = 0.2  # Reward for executing a high pass
        self.pass_completion_bonus = 0.1   # Extra reward for completing the pass

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
                      "precision_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Dynamic reward for managing a high pass:
            if 'game_mode' in o and o['game_mode'] == 6:  # Assuming game_mode 6 relates to high passes
                components["precision_pass_reward"][rew_index] += self.high_pass_score_reward

                # Additional bonus for completing the high pass:
                if 'ball_owned_player' in o and o['ball_owned_player'] == o['designated'] and \
                   o['ball_owned_team'] == rew_index % 2:
                    components["precision_pass_reward"][rew_index] += self.pass_completion_bonus

            # Updating the actual reward with the added component
            reward[rew_index] += components["precision_pass_reward"][rew_index]

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
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
