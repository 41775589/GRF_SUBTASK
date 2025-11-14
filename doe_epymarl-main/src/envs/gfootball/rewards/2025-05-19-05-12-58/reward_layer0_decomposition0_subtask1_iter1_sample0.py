import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds advanced defensive rewards with emphases on tackling, positioning, and transitions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.1
        self.positioning_reward = 0.2
        self.transition_reward = 0.3

    def reset(self):
        """Reset the environment and sticky counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current environment state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the environment state."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Customize reward to account for defensive tactics and transitions."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward),
            "transition_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for index, obs in enumerate(observation):
            # Reward tackling when near the ball and under opponent control
            if obs['ball_owned_team'] == 1 and np.linalg.norm(obs['left_team'][obs['active']] - obs['ball'][:2]) < 0.1:
                components["tackle_reward"][index] += self.tackle_reward
                reward[index] += components["tackle_reward"][index]

            # Reward positioning when the agent is between the ball and own goal
            ball_x_pos = obs['ball'][0]
            agent_x_pos = obs['left_team'][obs['active']][0]
            if obs['ball_owned_team'] == 1 and agent_x_pos < ball_x_pos:
                components["positioning_reward"][index] += self.positioning_reward
                reward[index] += components["positioning_reward"][index]

            # Reward efficient transitions from defense to attack
            if obs['ball_owned_team'] == 0 and np.linalg.norm(obs['left_team_direction'][obs['active']]) > 0.5:
                components["transition_reward"][index] += self.transition_reward
                reward[index] += components["transition_reward"][index]

        return reward, components

    def step(self, action):
        """Execute a step in the environment, processing custom rewards."""
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
