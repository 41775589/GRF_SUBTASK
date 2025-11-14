import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances defensive strategy through refined tactical rewards.
    It emphasizes coordinated defense, especially covering key zones and tackling/intercepting effectively.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.zone_defense_reward = 0.25  # Reward for maintaining formation in key defensive zones
        self.interception_tackle_reward = 0.3  # Reward for successful tackle or interception

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
        components = {
            "base_score_reward": reward.copy(),
            "zone_defense_reward": [0.0] * len(reward),
            "interception_tackle_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for idx, obs in enumerate(observation):
            player_pos = obs['left_team'][obs['active']]
            ball_pos = obs['ball'][:2]

            # Calculating a zone defense reward, emphasizing staying near the defensive key zones
            if player_pos[0] <= -0.5:  # Considered a key defensive zone near own goal
                components["zone_defense_reward"][idx] += self.zone_defense_reward

            # Reward for tackles and interceptions
            if obs['ball_owned_team'] == 1:  # If the ball is with the opponent
                distance_to_ball = np.linalg.norm(player_pos - ball_pos)
                if distance_to_ball < 0.1:  # Arbitrary threshold for 'close enough to tackle/intercept'
                    components["interception_tackle_reward"][idx] += self.interception_tackle_reward

            # Updating the reward with computed component values
            reward[idx] += components["zone_defense_reward"][idx] + components["interception_tackle_reward"][idx]

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
