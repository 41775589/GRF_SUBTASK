import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive skill reinforcement reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """ Reset sticky actions count and environment on episode start. """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """ Extract state from environment, including custom wrapper state if needed. """
        # Here you can add data extraction properties if needed.
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """ Set state of environment, including custom wrapper state if needed. """
        from_pickle = self.env.set_state(state)
        # Add state setting mechanism here if specific states need to be restored.
        return from_pickle

    def reward(self, reward):
        """ Calculate enhanced reward based on defensive interceptions and tackles. """
        observation = self.env.unwrapped.observation()  # Obtain game state
        if observation is None:
            return reward

        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward)}

        for idx, obs in enumerate(observation):
            if obs['ball_owned_team'] == 1:
                # Calculate player's distance to the ball, if the opponent controls the ball.
                own_player_position = obs['left_team'][obs['active']]
                ball_position = obs['ball'][:2]  # Ball's x, y position
                distance_to_ball = np.linalg.norm(own_player_position - ball_position)

                # Reward players getting closer to the ball (interception opportunities).
                inverted_distance = 1 / (distance_to_ball + 0.1)  # Add small constant to avoid division by zero

                # Decision: reward strategy based on commitment to defense
                components['defensive_reward'][idx] = 0.1 * inverted_distance

            # Update the total reward incorporating the defense-oriented reward component.
            reward[idx] += components['defensive_reward'][idx]

        return reward, components

    def step(self, action):
        """ Take an action using the overridden step method to inject our custom reward. """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
