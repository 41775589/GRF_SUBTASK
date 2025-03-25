import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive maneuvers and dynamic game adaptation."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Customize these parameters as needed
        self.offensive_reward_coefficient = 0.5
        self.game_mode_change_reward = 0.3

    def reset(self):
        # Reset sticky actions counter on each reset
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Load the state if necessary, currently we do not have a stored state
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        # Initialize reward components for reporting
        components = {"base_score_reward": reward.copy(),
                      "offensive_reward": [0.0] * len(reward),
                      "game_mode_change_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Give extra reward if your team possesses the ball
            if o['ball_owned_team'] == 1:
                components["offensive_reward"][rew_index] = self.offensive_reward_coefficient
                reward[rew_index] += components["offensive_reward"][rew_index]

            # Reward changes in the game mode, indicating dynamic game situations
            if o['game_mode'] != 0 and self.prev_game_mode[rew_index] == 0:
                components["game_mode_change_reward"][rew_index] = self.game_mode_change_reward
                reward[rew_index] += components["game_mode_change_reward"][rew_index]

            # Store current game mode to check for changes in the next step
            self.prev_game_mode[rew_index] = o['game_mode']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        # Add components to info for detailed logging
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Tracking sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
