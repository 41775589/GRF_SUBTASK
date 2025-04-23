import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive maneuvers reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the counter for sticky actions on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment with wrapper-specific data."""
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        """Set the state of the environment and wrapper-specific data."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, rewards):
        """Custom reward function focused on defensive plays."""
        obs = self.env.unwrapped.observation()
        base_score_reward = rewards
        defensive_reward = [0.0] * len(rewards)

        if obs is None:
            return rewards, {'base_score_reward': base_score_reward, 'defensive_reward': defensive_reward}

        for i, reward in enumerate(rewards):
            player_obs = obs[i]
            defense_bonus = 0
            active_player_y = player_obs['right_team'][player_obs['active']][1]
            
            # Reward for tackling: more reward closer to own goal to promote defense
            if player_obs['game_mode'] == 2:  # Assume game mode 2 is a defensive mode
                defense_bonus += 0.2 * (1 - abs(active_player_y))
            
            # Encourage use of slide action in dangerous situations
            if player_obs['sticky_actions'][9] == 1:
                defense_bonus += 0.1  # slide is assumed to be index 9 in sticky actions
            
            defensive_reward[i] = defense_bonus
            rewards[i] += defense_bonus

        return rewards, {'base_score_reward': base_score_reward, 'defensive_reward': defensive_reward}

    def step(self, action):
        """Apply actions, step the environment, and apply reward modification."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()

        # track sticky actions counts for debugging or further manipulation
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, ac in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = ac
        
        return obs, reward, done, info
