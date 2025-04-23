import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive actions and strategic positioning for counterattacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Modify rewards for defensive strategy
        self.defensive_rewards = {
            "steal_ball": 1.0,     # Reward for stealing the ball
            "block_shot": 0.5,     # Reward for blocking an opponent's shot
            "clear_ball": 0.3,     # Reward for clearing the ball from near the goal area
            "successful_tackle": 0.7  # Reward for successful tackles
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_rewards": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate rewards based on defensive plays
            is_defensive_situation = (o['game_mode'] in [3, 4, 5, 6])  # FreeKick, Corner, ThrowIn, Penalty

            if o['ball_owned_team'] == 0 and is_defensive_situation:
                # Ball is owned by controlled team during a game mode that requires defense
                defensive_reward = sum(self.defensive_rewards[action] for action in o['sticky_actions'] if action in self.defensive_rewards)
                components['defensive_rewards'][rew_index] = defensive_reward
                reward[rew_index] += defensive_reward

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
