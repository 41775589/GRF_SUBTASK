import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful tackles and penalizes losing control during tackles."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # for tracking sticky actions frequency
        self.previous_game_mode = None
        self.previous_ball_owned_team = None

    def reset(self):
        """Reset the sticky action counter, previous game mode, and previous ball ownership."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_game_mode = None
        self.previous_ball_owned_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include wrapper-specific state elements."""
        state = self.env.get_state(to_pickle)
        state['previous_game_mode'] = self.previous_game_mode
        state['previous_ball_owned_team'] = self.previous_ball_owned_team
        return state

    def set_state(self, state):
        """Restore wrapper-specific state elements."""
        self.previous_game_mode = state['previous_game_mode']
        self.previous_ball_owned_team = state['previous_ball_owned_team']
        return self.env.set_state(state)

    def reward(self, reward):
        """Adjust the reward based on tackle success and ball possession specifics."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "penalty_for_loss": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            if obs['game_mode'] != self.previous_game_mode and obs['game_mode'] == 3:  # FreeKick
                # Tackle was likely successful just before FreeKick mode
                components["tackle_reward"][idx] = 0.5
                reward[idx] += components["tackle_reward"][idx]

            if self.previous_ball_owned_team is not None and obs['ball_owned_team'] != self.previous_ball_owned_team and obs['ball_owned_team'] == -1:
                # Ball was lost after being owned
                components["penalty_for_loss"][idx] = -0.3
                reward[idx] += components["penalty_for_loss"][idx]

            # Updating for next call
            self.previous_game_mode = obs['game_mode']
            self.previous_ball_owned_team = obs['ball_owned_team']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Populate info dict with total and components of rewards for debugging/analysis
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions info
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
