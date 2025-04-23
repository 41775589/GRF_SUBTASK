import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for mastering long passes with accuracy.
    This involves checking the ball's travel distance and precision of landing.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_reward = 0.2
        self.precision_factor = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0, 0.0]}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            ball_position = o['ball']
            ball_direction = o['ball_direction']
            ball_distance = np.linalg.norm(ball_direction[:2])

            # Reward based on ball travel distance
            if ball_distance > 0.5:  # Assuming long passes are those that move the ball significantly
                components["long_pass_reward"][idx] = self.long_pass_reward

                # Additional reward for accuracy: checking if it nears a teammate
                bonus_for_accuracy = 0
                team_pos = o['right_team'] if o['ball_owned_team'] == 1 else o['left_team']
                for teammate in team_pos:
                    if np.linalg.norm(teammate - ball_position[:2]) < 0.1:
                        bonus_for_accuracy = self.precision_factor
                        break
                components["long_pass_reward"][idx] += bonus_for_accuracy
                reward[idx] += components["long_pass_reward"][idx]

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
