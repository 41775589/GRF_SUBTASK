import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies the rewards to focus on maintaining ball control under pressure, 
    making strategic plays, and distributing the ball effectively across the field."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pressure_coefficient = 0.1
        self.strategy_coefficient = 0.2
        self.distribution_coefficient = 0.3

    def reset(self):
        """Reset the environment and counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """Customize rewards to focus on ball control, strategic plays, and effective distribution."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pressure_reward": [0.0] * len(reward),
            "strategic_play_reward": [0.0] * len(reward),
            "distribution_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            # Pressure handling: reward based on maintaining possession under close defense
            if obs['ball_owned_team'] == 0 and np.any(np.linalg.norm(obs['opponent_team'] - obs['ball'], axis=1) < 0.1):
                components["pressure_reward"][i] = self.pressure_coefficient

            # Strategic plays: reward for switching play to less crowded areas
            if obs['ball_owned_team'] == 0:
                team_distribution = np.mean(obs['teammate_team'], axis=0)
                ball_location = obs['ball'][:2]
                if np.abs(ball_location - team_distribution).max() > 0.5:
                    components["strategic_play_reward"][i] = self.strategy_coefficient
            
            # Effective distribution: reward for passes leading to significant forward movement
            prev_pos = self.env.unwrapped.previous_ball_position
            if prev_pos is not None and obs['ball_owned_team'] == 0 and (obs['ball'][0] - prev_pos[0]) > 0.1:
                components["distribution_reward"][i] = self.distribution_coefficient

            # Combine rewards
            reward[i] += sum([
                components["pressure_reward"][i],
                components["strategic_play_reward"][i],
                components["distribution_reward"][i]
            ])

        return reward, components

    def step(self, action):
        """Step the environment, apply reward transformation, and return new state."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs is not None:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    self.sticky_actions_counter[i] = action
        return obs, reward, done, info
