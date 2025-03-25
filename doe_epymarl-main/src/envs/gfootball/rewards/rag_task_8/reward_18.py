import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for quick decision-making and efficient ball handling for counter-attacks."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No internal state to reset apart from base RewardWrapper
        return from_pickle

    def reward(self, reward):
        """Modify the reward based on possession regain and quick counter-attack initiation"""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Initially, we do not change the reward.
            checkpoint_reward = 0.0

            # We encourage quick ball handling and advancing the ball towards the opponent's goal after regaining possession.
            # Checking if the team just regained possession (ball_owned_team changes to 0 from -1).
            if o['ball_owned_team'] == 0 and o['game_mode'] == 0:  # Normal game mode
                ball_owner = o['ball_owned_player']
                if ball_owner == o['active']:  # Check if the controlled player is the one with the ball.
                    # Reward based on moving forward quickly towards the opponent's goal (opponent goal is at x=1)
                    ball_pos = o['ball'][0]
                    checkpoint_reward += (1 - abs(ball_pos)) * 0.5  # Increasing reward as the player moves forward with the ball.

            # Aggregating the rewards
            total_reward = reward[rew_index] + checkpoint_reward
            reward[rew_index] = total_reward
            components.setdefault("checkpoint_reward", []).append(checkpoint_reward)

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
