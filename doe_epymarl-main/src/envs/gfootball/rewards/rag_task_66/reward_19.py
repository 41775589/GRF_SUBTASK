import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward for mastering short passes under defensive pressure, focusing on ball retention and effective distribution."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_rewards = {}
        self.retention_reward_weight = 0.1
        self.distribution_reward_weight = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.passing_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.passing_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "retention_reward": [0.0] * len(reward),
                      "distribution_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward successful passes under pressure.
            if (o['game_mode'] == 2 and  # Assuming game_mode 2 is close passing play
                o['ball_owned_team'] == o['active'] and
                np.any(o['sticky_actions'][0:2]) and  # Assuming indices 0 and 1 are pass actions
                not o['left_team_yellow_card'][o['active']] and  # No yellow card for active player
                not o['right_team_yellow_card'][o['active']]):  # No yellow card for active player

                components["retention_reward"][rew_index] = self.retention_reward_weight
                reward[rew_index] += self.retention_reward_weight

            # Reward spreading the ball under pressure.
            if (o['game_mode'] == 2 and
                o['ball_direction'][0] > 0.1):  # Assuming moving the ball forward is good
                components["distribution_reward"][rew_index] = self.distribution_reward_weight
                reward[rew_index] += self.distribution_reward_weight

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
