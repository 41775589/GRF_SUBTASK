import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for successful standing tackles and ball control without penalties."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_ownership = -1  # Track ball ownership to detect tackles
        self.tackle_reward = 0.5  # Reward for successfully taking possession

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_ownership = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['previous_ball_ownership'] = self.previous_ball_ownership
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.previous_ball_ownership = from_pickle['previous_ball_ownership']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Loop over agents to compute additional rewards
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if ball possession has changed to the active player
            if ('ball_owned_team' in o and o['ball_owned_team'] != -1 and 
                o['ball_owned_team'] == o['active'] and
                self.previous_ball_ownership != o['active']):
                # Provide reward if ball was tackled successfully without a foul
                if ('game_mode' in o and o['game_mode'] == 0):  # Normal gameplay
                    components['tackle_reward'][rew_index] = self.tackle_reward
                    reward[rew_index] += self.tackle_reward

            # Update previous ball ownership
            self.previous_ball_ownership = o['ball_owned_team']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        if obs is not None:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
