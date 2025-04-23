import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive strategies reward for efficient tackles, movement control, and passing under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Actions tracking
        self._efficiency_coefficient = 0.05  # Coefficient to adjust the reward sensitivity to efficiency

    def reset(self):
        """Resets the environment and the action counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state of the wrapper."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Augments reward based on defensive actions and tactics."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_efficiency_reward": [0.0] * len(reward),
                      "movement_control_reward": [0.0] * len(reward),
                      "pressured_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index, obs in enumerate(observation):
            if obs['game_mode'] != 0:
                # Focusing rewards only during normal play
                continue

            # Reward for efficient tackling
            if obs['sticky_actions'][6] and obs['ball_owned_team'] == 0:  # Assume index 6 is a tackle action
                components['tackle_efficiency_reward'][rew_index] += self._efficiency_coefficient * 2
            
            # Movement control: reward players for staying in position when not owning the ball
            if obs['ball_owned_team'] == 1 and not any(obs['sticky_actions'][:4]):  # No movement actions
                components['movement_control_reward'][rew_index] += self._efficiency_coefficient

            # Pressured pass: applying pressure when the opposition tries to make a pass
            if obs['ball_owned_team'] == 1 and obs['game_mode'] in [1, 3]:  # KickOff or FreeKick
                components['pressured_pass_reward'][rew_index] += self._efficiency_coefficient * 1.5

            # Calculate total reward for this agent
            total_agent_reward = sum([
                reward[rew_index],
                components['tackle_efficiency_reward'][rew_index],
                components['movement_control_reward'][rew_index],
                components['pressured_pass_reward'][rew_index]
            ])
            reward[rew_index] = total_agent_reward

        return reward, components

    def step(self, action):
        """Steps through the environment, applying the reward modifications."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
