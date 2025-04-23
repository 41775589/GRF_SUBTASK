import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on enhancing strategic positioning, emphasizing better defensive responses and quick transitions from defense to counterattack."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize parameters for defensive positioning reward
        self.defensive_positioning_reward = 0.05
        self.counterattack_transition_reward = 0.1
        self.ball_recovery_reward = 0.15

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward),
                      "counterattack_transition_reward": [0.0] * len(reward),
                      "ball_recovery_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for maintaining useful defensive positions
            if o['ball_owned_team'] != 1:  # Ball not owned by opponents
                average_depth = np.mean([pos[0] for pos in o['left_team']])
                defensive_bonus = self.defensive_positioning_reward * max(0, 1 - average_depth)
            else:
                defensive_bonus = 0

            # Reward for quick transition from defense to counterattack
            if o['game_mode'] in {0, 2, 4, 6} and o['ball_owned_team'] == 0:  # Regular play or set piece and ball owned by team
                transition_bonus = self.counterattack_transition_reward
            else:
                transition_bonus = 0

            # Reward for recovering the ball from a defensive position
            if o['ball_owned_team'] == 0 and o['previous_ball_owned_team'] == 1:
                recovery_bonus = self.ball_recovery_reward
            else:
                recovery_bonus = 0

            # Update rewards components
            components["defensive_positioning_reward"][rew_index] = defensive_bonus
            components["counterattack_transition_reward"][rew_index] = transition_bonus
            components["ball_recovery_reward"][rew_index] = recovery_bonus

            # Calculate total reward
            reward[rew_index] += defensive_bonus + transition_bonus + recovery_bonus

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
