import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards efficient ball handling and quick decision-making for initiating counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.ball_recovery_counter = 0
        self.quick_counterattack_bonus = 0.5
        self.recovery_decay = 0.9
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.ball_recovery_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'ball_recovery_counter': self.ball_recovery_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_recovery_counter = from_pickle['CheckpointRewardWrapper']['ball_recovery_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counterattack_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Reward for ball recovery in the defensive half and quick transition to offensive half
            if o['ball_owned_team'] == 0 and o['ball'][0] < 0:
                self.ball_recovery_counter += 1

            if o['ball_owned_team'] == 0 and o['ball'][0] > 0:
                reward_rate = self.quick_counterattack_bonus * (self.recovery_decay ** self.ball_recovery_counter)
                components["counterattack_reward"][rew_index] = reward_rate
                reward[rew_index] += reward_rate

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
