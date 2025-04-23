import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive strategies including shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize coefficients for each action that contributes to skill mastery
        self.passing_coefficient = 0.2
        self.shooting_coefficient = 0.3
        self.dribbling_coefficient = 0.1

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Saves the wrapper state information into the pickle dictionary."""
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the wrapper state from the given state dictionary."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Computes a modified reward that emphasizes successful offensive maneuvers."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shoot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Increment reward for dribbling when 'action_dribble' is active
            if o['sticky_actions'][9] == 1: # action_dribble
                components['dribble_reward'][rew_index] = self.dribbling_coefficient
                reward[rew_index] += components['dribble_reward'][rew_index]
            
            # Reward for successful passes, considering a change in ball possession among teammates
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] != o['active'] and (o['game_mode'] == 1 or o['game_mode'] == 6):
                # Assumes game_mode 1 or 6 involves successful passing mechanics (KickOff or Penalty which might simulate key passes)
                components['pass_reward'][rew_index] = self.passing_coefficient
                reward[rew_index] += components['pass_reward'][rew_index]
            
            # Reward for shooting towards the goal 
            # Simplified by assuming a shot when the ball is directed significantly towards the opponent's goal zone.
            if o['ball_owned_team'] == 0 and np.linalg.norm(o['ball_direction'][:2]) > 0.5:
                components['shoot_reward'][rew_index] = self.shooting_coefficient
                reward[rew_index] += components['shoot_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Executes a step in the environment, augments it with reward calculation based on stored policies."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Update sticky actions counter based on the current actions
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_activated in enumerate(agent_obs['sticky_actions']):
                if action_activated:
                    self.sticky_actions_counter[i] += 1
        return observation, reward, done, info
