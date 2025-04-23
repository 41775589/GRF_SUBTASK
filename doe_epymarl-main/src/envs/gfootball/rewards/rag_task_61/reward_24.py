import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards focused on enhancing team synergy during possession changes.
    Rewards are given for precise timing and strategic positioning during both offensive and defensive actions.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        # Reset sticky actions tracking and environment
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Save the current state of the wrapper along with the environment state
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        # Restore the state of the wrapper along with the environment state
        from_pickle = self.env.set_state(state)
        # Nothing specific in this example wrapper's state to restore
        return from_pickle

    def reward(self, reward):
        # Customize the reward function to enhance team synergy
        
        observation = self.env.unwrapped.observation()
        
        # Initialize reward components for each agent
        components = {
            "base_score_reward": reward.copy(),
            "possession_change_bonus": [0.0, 0.0]
        }
        
        if observation is None:
            return reward, components
        
        for idx, obs in enumerate(observation):
            # Define a bonus for possession change with strategic repositioning and coordination
            if obs['game_mode'] in {2, 3, 4, 5, 6} and self._has_possession_changed(obs):
                # Adding a flat bonus for effective strategy after possession change
                components['possession_change_bonus'][idx] = 0.5
                reward[idx] += components['possession_change_bonus'][idx]

        return reward, components

    def step(self, action):
        # Perform an environment step and augment reward calculation
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        return observation, reward, done, info

    def _has_possession_changed(self, obs):
        # Check if possession has changed dramatically to encourage strategic play:
        opp_team = 0 if obs['ball_owned_team'] == 1 else 1
        return obs['ball_owned_team'] == opp_team
