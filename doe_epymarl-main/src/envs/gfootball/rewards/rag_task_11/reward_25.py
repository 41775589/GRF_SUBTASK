import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive maneuvers and precision-based finishing."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.goal_approach_reward = 0.05
        self.precision_finishing_reward = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "goal_approach_reward": [0.0] * len(reward),
                      "precision_finishing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] != 1:
                # In play (not in set piece like corner, free-kick)
                # Encourage forward movement towards the goal
                ball_x = o['ball'][0]
                goal_x = 1.0 if o['ball_owned_team'] == 0 else -1.0

                # Reward moving the ball to final third of the pitch relative to their goal direction
                if np.sign(ball_x) == np.sign(goal_x) and abs(ball_x) > 0.66:
                    components["goal_approach_reward"][rew_index] = self.goal_approach_reward
                    reward[rew_index] += components["goal_approach_reward"][rew_index]

                # Reward precision in ball control when close to goal
                if np.sign(ball_x) == np.sign(goal_x) and abs(ball_x) > 0.9:
                    components["precision_finishing_reward"][rew_index] = self.precision_finishing_reward
                    reward[rew_index] += components["precision_finishing_reward"][rew_index]

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
