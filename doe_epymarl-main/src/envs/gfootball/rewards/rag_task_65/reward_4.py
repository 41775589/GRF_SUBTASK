import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a scenario-based reward focusing on shooting and passing accuracy."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.shooting_precision = 0.1
        self.passing_precision = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapperData'] = {'ShootingPrecision': self.shooting_precision,
                                                    'PassingPrecision': self.passing_precision}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shooting_precision = from_pickle['CheckpointRewardWrapperData']['ShootingPrecision']
        self.passing_precision = from_pickle['CheckpointRewardWrapperData']['PassingPrecision']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_precision": [0.0] * len(reward),
            "passing_precision": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Evaluate shooting: check if this is a goal kick
            if o['game_mode'] == 6:  # Game mode for Penalty
                components["shooting_precision"][rew_index] = self.shooting_precision
                reward[rew_index] += self.shooting_precision

            # Evaluate passing: check if the sticky action for passing was just activated
            if 'sticky_actions' in o:
                pass_action = o['sticky_actions'][1]  # Index for pass action
                if self.sticky_actions_counter[rew_index, 1] == 0 and pass_action == 1:
                    components["passing_precision"][rew_index] = self.passing_precision
                    reward[rew_index] += self.passing_precision
                self.sticky_actions_counter[rew_index, :] = o['sticky_actions']

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
