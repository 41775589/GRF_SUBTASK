import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive and midfield control tasks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define weights or coefficients for specific behaviors
        self.defensive_play_weight = 0.2
        self.midfield_control_weight = 0.15
        self.ball_stealing_weight = 0.1
        self.positional_rewards = {}

    def reset(self):
        """Reset the environment and clear any positional rewards."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positional_rewards = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store additional state for checkpoints."""
        to_pickle['positional_rewards'] = self.positional_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore additional state for checkpoints."""
        from_pickle = self.env.set_state(state)
        self.positional_rewards = from_pickle['positional_rewards']
        return from_pickle

    def reward(self, reward):
        """Custom reward function focused on defensive play and midfield control."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_play_reward": [0.0] * len(reward),
            "midfield_control_reward": [0.0] * len(reward),
            "ball_stealing_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_team = o['ball_owned_team']

            # Rewards for maintaining defensive positions or regaining control in the defense area
            if ball_owned_team == 0 and o['left_team'][o['active']][0] < -0.3:
                components["defensive_play_reward"][rew_index] = self.defensive_play_weight

            # Rewards for control in the midfield area
            if -0.2 <= o['left_team'][o['active']][0] <= 0.2:
                components["midfield_control_reward"][rew_index] = self.midfield_control_weight

            # Rewards for stealing the ball in the opponent's possession
            if ball_owned_team == 1 and o['right_team'][o['active']][0] < 0:
                components["ball_stealing_reward"][rew_index] = self.ball_stealing_weight

            # Calculate the total reward for this agent
            reward[rew_index] += (
                components["defensive_play_reward"][rew_index] +
                components["midfield_control_reward"][rew_index] +
                components["ball_stealing_reward"][rew_index]
            )
        
        return reward, components

    def step(self, action):
        """Step the environment, applying the new reward scheme."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
