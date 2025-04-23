import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies reward based on positional play,
    encouraging strategic positioning, lateral and backward movements,
    and quick transitions from defense to attack.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.array([0., 0., 0.])
        self.transition_reward = 0.05
        self.defensive_play_reward = 0.02

    def reset(self):
        """
        Reset the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.array([0., 0., 0.])
        return self.env.reset()

    def reward(self, reward):
        """
        Customize the reward to strengthen strategic gameplay.
        """
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {"base_score_reward": reward}

        components = {"base_score_reward": reward.copy(),
                      "transition_reward": [0.0, 0.0],
                      "defensive_play_reward": [0.0, 0.0]}
        ball_position = observation[0]['ball']

        for idx in range(2):
            o = observation[idx]
            ball_owned_team = o['ball_owned_team']
            if ball_owned_team == 1:
                continue  # Reward only if ball is owned by the right team

            # Calculate transition reward for moving the ball forward efficiently
            if np.linalg.norm(self.previous_ball_position - ball_position) > 0.05:
                if ball_position[0] > self.previous_ball_position[0]:  # Forward movement
                    components['transition_reward'][idx] = self.transition_reward
                    reward[idx] += components['transition_reward'][idx]

            # Defensive positioning reward enhancing backward and lateral movements
            if o['right_team_roles'][o['active']] in [1, 2, 3, 4]:  # Defensive roles
                if abs(ball_position[1]) < abs(self.previous_ball_position[1]):  # Better lateral coverage
                    components['defensive_play_reward'][idx] = self.defensive_play_reward
                    reward[idx] += components['defensive_play_reward'][idx]

        # Update previous ball position
        self.previous_ball_position = ball_position
        return reward, components

    def step(self, action):
        """
        Execute a step in the environment, recording changes and issuing modified rewards.
        """
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
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
