import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adjusts rewards based on defensive behavior, focusing on tackles without fouling."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the environment and clear counters related to sticky actions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Get the state of the environment including the custom modifications in the wrapper."""
        pickle_dict = self.env.get_state(to_pickle)
        return pickle_dict

    def set_state(self, state):
        """Set the state of the environment including the custom modifications in the wrapper."""
        env_state = self.env.set_state(state)
        return env_state
    
    def reward(self, reward):
        """Adjust rewards based on defensive actions performed correctly without causing fouls."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components

        # Update components dictionary for detailed reward components tracking
        components['tackle_reward'] = [0.0] * len(reward)
        components['foul_penalty'] = [0.0] * len(reward)

        for idx, (obs, rew) in enumerate(zip(observation, reward)):
            # Incorporate logic for verified tackling
            if obs['game_mode'] in [3, 4] and obs['ball_owned_player'] == obs['active']:
                # Positive feedback for successful tackles in free kick or corner situations
                components['tackle_reward'][idx] = 0.5
                reward[idx] += components['tackle_reward'][idx]

            # Update to penalize fouls
            if obs['game_mode'] in [6] and (obs['right_team_yellow_card'].any() or obs['left_team_yellow_card'].any()):
                # Penalize players causing fouls leading to penalties
                components['foul_penalty'][idx] = -0.3
                reward[idx] += components['foul_penalty'][idx]
        
        return reward, components

    def step(self, action):
        """Proceed with the normal step function but modify the `reward` to include the custom components."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        
        # Include the detailed components in the info dictionary for better transparency
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        # Count sticky actions in the current state after stepping
        self.sticky_actions_counter.fill(0)
        current_obs = self.env.unwrapped.observation()
        for agent_obs in current_obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        
        return observation, reward, done, info
