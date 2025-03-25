import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper focused on promoting behaviors typical for a 'sweeper' role—clearing balls, making last-man tackles,
    and successful recoveries, specifically enhancing learning in these areas."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_rewards = 0.1
        self.tackle_rewards = 0.2
        self.recovery_rewards = 0.15
        self.ball_clearance_zone_y = [-0.42, 0.42]  # Y range to consider as the defensive zone to clear the ball

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_reward_wrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward),
                      "recovery_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if it's in the defensive zone and clear the ball
            if o['ball'][1] >= self.ball_clearance_zone_y[0] and o['ball'][1] <= self.ball_clearance_zone_y[1]:
                if o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0:  # Assuming left team is ours
                    components["clearance_reward"][rew_index] = self.clearance_rewards
                    reward[rew_index] += components["clearance_reward"][rew_index]

            # Reward tackles - last man standing context assumed by proximity to goal x-pos < -0.7
            if (o['game_mode'] == 3) and (o['left_team'][o['active']][0] < -0.7):
                components["tackle_reward"][rew_index] = self.tackle_rewards
                reward[rew_index] += components["tackle_reward"][rew_index]

            # Reward for recoveries: impacting play after loss of possession
            if ('game_mode' in o) and (o['game_mode'] == 0):
                if o['ball_owned_team'] == 1 and o['left_team'][o['active']][0] < -0.5:
                    components["recovery_reward"][rew_index] = self.recovery_rewards
                    reward[rew_index] += components["recovery_reward"][rew_index]

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
