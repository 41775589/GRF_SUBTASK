import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic offensive maneuver reward to the base football environment."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._offensive_progress_bonus = 0.05  # Bonus reward for offensive progress
        self._ball_progress_cache = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_progress_cache = []
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'offensive_bonus': [0.0] * 2}
        
        for i, o in enumerate(observation):
            ball_pos = o['ball'][:2]
            offensive_progress = self._evaluate_offensive_progress(ball_pos, i)
            components['offensive_bonus'][i] = offensive_progress
            reward[i] += components['offensive_bonus'][i]
        
        return reward, components

    def _evaluate_offensive_progress(self, ball_pos, player_idx):
        """Evaluate the ball's progress towards the opponent's goal for reward purposes."""
        
        if len(self._ball_progress_cache) <= player_idx:
            self._ball_progress_cache.append(ball_pos[0])
            return 0
        
        progress = 0
        # Reward forward movement towards the opponent's goal (which is at x=1)
        if ball_pos[0] > self._ball_progress_cache[player_idx]:
            progress += self._offensive_progress_bonus
        
        self._ball_progress_cache[player_idx] = ball_pos[0]
        return progress

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
