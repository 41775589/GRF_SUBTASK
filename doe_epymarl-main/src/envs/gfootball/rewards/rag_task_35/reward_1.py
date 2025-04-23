import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on strategic positioning and effective transitioning between defense and attack."""

    def __init__(self, env):
        super().__init__(env)
        self.base_reward_coef = 1.0
        self.positioning_reward_coef = 0.2
        self.transitioning_reward_coef = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper_sticky_actions'], dtype=int)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0] * len(reward),
            "transitioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_reward = components["base_score_reward"][rew_index]

            # Positioning Reward - Encourages strategic position maintaining
            x, y = o['ball'][0], o['ball'][1]  # Uses the ball's position
            player_x, player_y = o['left_team'][o['active']][:2] # Active player's position

            # Encourages players to maintain a strategic position relative to ball
            if abs(player_x - x) < 0.2:  # Good horizontal alignment with the ball
                components["positioning_reward"][rew_index] = self.positioning_reward_coef

            # Transitioning Reward - Motivates switching between defense and attack strategies
            if o['game_mode'] == 0:  # Normal gameplay (0 is normal game mode)
                player_direction = o['left_team_direction'][o['active']]
                ball_direction = o['ball_direction'][:2]
                dot_product = np.dot(player_direction, ball_direction)  # Cosine similarity of movement direction
                if dot_product > 0:  # Moving in the same direction as the ball
                    components["transitioning_reward"][rew_index] = self.transitioning_reward_coef
            
            # Compose final reward for this agent
            reward[rew_index] = (current_reward +
                                 components["positioning_reward"][rew_index] +
                                 components["transitioning_reward"][rew_index])
        
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
