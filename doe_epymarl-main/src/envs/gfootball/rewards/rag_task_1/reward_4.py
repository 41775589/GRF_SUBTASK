import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds dynamic offensive maneuver rewards based on game phases."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
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

        # Initialize the components dictionary
        components = {
            "base_score_reward": reward.copy(),
            "quick_attack_bonus": [0.0] * len(reward),
            "possession_bonus": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]

            # Quick attack bonus when transitioning into attacking game modes quickly
            if obs['game_mode'] in [2, 3, 4, 5, 6]:  # These modes involve set pieces or interruptions
                components["quick_attack_bonus"][rew_index] = 0.2

            # Possession bonus if the ball is owned by the player's team and moving forward
            if obs['ball_owned_team'] == 1 and obs['ball_direction'][0] > 0:  # Assuming team 1 attacks right
                components["possession_bonus"][rew_index] = 0.1

            # Update cumulative reward
            reward[rew_index] += components["quick_attack_bonus"][rew_index] + components["possession_bonus"][rew_index]

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
