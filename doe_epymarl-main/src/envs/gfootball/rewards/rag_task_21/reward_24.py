import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for defensive actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.intercept_reward = 1.0
        self.positioning_reward = 0.5
        self.ball_close_reward = 0.3
        self.defensive_zones = [
            [-1, -0.42],  # Left bottom corner
            [-0.5, -0.42],  # Left center
            [0, 0],  # Center field
            [0.5, 0.42],  # Right center
            [1, 0.42]  # Right top corner
        ]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "intercept_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward),
                      "ball_close_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                # Reward interception if applicable
                components["intercept_reward"][rew_index] = self.intercept_reward
                reward[rew_index] += components["intercept_reward"][rew_index]
            
            # Calculate distance to defensive strategic points
            for zone in self.defensive_zones:
                dist = np.linalg.norm(np.array(o['left_team'][0][:2]) - np.array(zone))
                if dist < 0.1:  # Player is in a good defensive position
                    components["positioning_reward"][rew_index] += self.positioning_reward

            # Ball proximity increases the player's awareness
            ball_pos = np.array(o['ball'][:2])
            player_pos = np.array(o['left_team'][o['active']][:2])
            if np.linalg.norm(ball_pos - player_pos) < 0.2:
                components["ball_close_reward"][rew_index] += self.ball_close_reward

            reward[rew_index] += (components["positioning_reward"][rew_index] +
                                  components["ball_close_reward"][rew_index])
        
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
