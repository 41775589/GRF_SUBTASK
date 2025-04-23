import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """Wrapper that modifies the reward function to focus on quick counterattacks and long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_counter = 0
        self.transition_counter = 0
        self.long_pass_reward_coefficient = 1.0
        self.transition_reward_coefficient = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_counter = 0
        self.transition_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'long_pass_counter': self.long_pass_counter,
            'transition_counter': self.transition_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.long_pass_counter = from_pickle['CheckpointRewardWrapper']['long_pass_counter']
        self.transition_counter = from_pickle['CheckpointRewardWrapper']['transition_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward for long passes (Assuming long pass is defined by a significant increase in the x-direction of the ball)
            if o['ball_owned_team'] == o['active'] and np.linalg.norm(o['ball_direction'][:2]) > 0.05:
                if np.abs(o['ball_direction'][0]) > np.abs(o['ball_direction'][1]) * 2:
                    components["long_pass_reward"][rew_index] = self.long_pass_reward_coefficient
                    reward[rew_index] += components["long_pass_reward"][rew_index]
                    self.long_pass_counter += 1

            # Reward for quick transition from defense to attack
            # Assuming transition from defense to attack involves moving the ball from a negative to a positive x position rapidly
            if o['ball_owned_team'] == 0 and o['ball'][0] < -0.5 and np.any(o['right_team'][:, 0] > 0):
                if np.any(o['right_team'][:, 0] - o['right_team_direction'][:, 0] < 0):
                    components["transition_reward"][rew_index] = self.transition_reward_coefficient
                    reward[rew_index] += components["transition_reward"][rew_index]
                    self.transition_counter += 1

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
