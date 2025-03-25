import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive skill-based reward focused on actions like sliding,
    stop-dribble, and stop-moving to effectively block opponents' attacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track the sticky actions
        
    def reset(self):
        """Reset everything for the start of a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieve the state for serialization."""
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state from deserialization."""
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Custom reward function focusing on defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_actions_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            # Rewarding sliding action if player is near an opponent with ball
            if obs['sticky_actions'][9] == 1 and 
                obs['ball_owned_team'] == 1 - obs['active']:  # assuming '1' is the opposite team
                # Compute opponent ball distance for additional reward
                opponent_ball_distance = np.linalg.norm(obs['ball'] - obs['right_team'][obs['ball_owned_player']])
                if opponent_ball_distance < 0.3:  # reward proximity defense
                    components['defensive_actions_reward'][rew_index] = 0.1

            # Reward stopping the dribble and movement near opponent's attacking players
            if obs['sticky_actions'][9] == 0 and obs['ball_owned_team'] == 1:  # ball with the opponent
                opponent_positions = obs['right_team']
                player_position = obs['left_team'][obs['active']]
                distances = np.linalg.norm(opponent_positions - player_position, axis=1)
                close_opponents = np.sum(distances < 0.2)
                components['defensive_actions_reward'][rew_index] += 0.05 * close_opponents

            # Update the reward
            reward[rew_index] += components['defensive_actions_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Retrieve info, adjust reward, and report."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
