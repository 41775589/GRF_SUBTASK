import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on executing high passes and controlling their trajectory and power. This focuses on precision and situational use of high passes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_accuracy_reward = 0.1
        self.power_assessment_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "high_pass_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage executing high passes more accurately.
            if 'game_mode' in o and o['game_mode'] == 6:  # assuming game_mode 6 is related to passing
                components["high_pass_reward"][rew_index] += self.high_pass_accuracy_reward
                reward[rew_index] += components["high_pass_reward"][rew_index]

            # Reward power control for long distances.
            if o['ball_owned_team'] in [0, 1] and 'ball_direction' in o:
                z_component = o['ball_direction'][2]  # Vertical component of the ball's motion
                if z_component > 0.5:  # Assuming that 0.5 might be a threshold for "high" passes
                    components["high_pass_reward"][rew_index] += self.power_assessment_reward
                    reward[rew_index] += components["high_pass_reward"][rew_index]

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
            if 'sticky_actions' in agent_obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
