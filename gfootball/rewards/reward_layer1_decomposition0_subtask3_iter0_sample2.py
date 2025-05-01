import gym
import numpy as np


class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on close-range attacks, dribbling efficiency, and quick decision-making in front of the opponent's goal."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Tracking sticky actions

    def reset(self):
        """
        Resets the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Saves the current state of the wrapper along with its environment state.
        """
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        """
        Restores the state of the wrapper and its environment from the saved state.
        """
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Modifies the reward based on close-range attacking metrics.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.copy(reward)}

        if observation is None:
            return reward, components

        adjusted_rewards = np.zeros_like(reward)

        for index, obs in enumerate(observation):
            # Use position close to the opponent's goal to increase reward
            if obs['ball_owned_team'] == 0:  # Assuming 0 is the team of the agent
                ball_y = abs(obs['ball'][1])  # Y-position of the ball
                # Closer to the goal line (Y near 0) yields higher reward
                close_range_bonus = max(0, (0.042 - ball_y) / 0.042)
                adjusted_rewards[index] += close_range_bonus * 0.1  # scale factor for closeness

                # Encourage shooting at close range to the goal by checking game mode
                if 'game_mode' in obs and obs['game_mode'] == 0:  # 0 for normal gameplay
                    adjusted_rewards[index] += 0.2  # bonus for shooting in normal play mode near the goal

                # Reward dribbling actions (assumption: dribble index=9 in sticky actions)
                dribbling = obs['sticky_actions'][9]
                adjusted_rewards[index] += dribbling * 0.05  # reward for dribbling near the goal

        # Combine rewards and components
        final_rewards = reward + adjusted_rewards
        components.update({
            "close_range_bonus": adjusted_rewards,
        })

        return final_rewards, components

    def step(self, action):
        """
        Steps through the environment, applying reward modifications.
        """
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

        return observation, reward, done, info