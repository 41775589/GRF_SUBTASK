import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive training reward based on defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper_sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Base reward is the untouched reward from the original environment
        components = {"base_score_reward": reward.copy()}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        # Defensive reward calculations
        defensive_reward = [0.0] * len(reward)
        for i, o in enumerate(observation):
            # Encourage players to stay close to the opponent having the ball, for interception or tackling
            if o['ball_owned_team'] == 1:  # If the ball is with the right team (opponents)
                # Calculate distance to the ball
                ball_x, ball_y = o['ball'][:2]
                player_x, player_y = o['left_team'][o['active']][:2]
                distance_to_ball = np.sqrt((ball_x - player_x) ** 2 + (ball_y - player_y) ** 2)

                # Smaller distances get a higher reward
                defensive_reward[i] = max(0.1, 1 - distance_to_ball)
            
            # Deduct points for fouls (observable by yellow cards)
            if 'left_team_yellow_card' in o and o['left_team_yellow_card'][o['active']]:
                defensive_reward[i] -= 0.5

            # Normalize the defensive reward component to be between 0 and 1
            defensive_reward[i] = np.clip(defensive_reward[i], 0, 1)

            # Scaling the defensive reward with a factor to weight its importance
            reward[i] += 0.5 * defensive_reward[i]  # This can be tuned based on importance

        components['defensive_reward'] = defensive_reward
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Add component values and final reward values to info for analysis
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update the sticky actions counter for analytical purposes
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
