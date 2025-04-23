import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive play and quick transition into counterattacks."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = np.linspace(-1, 0, 5)  # Positions on the left half towards the own goal
        self.collected_positions = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.collected_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.collected_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.collected_positions = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        # Component-wise reward initialization
        components = {
            "base_score_reward": reward.copy(),
            "defensive_position_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for idx in range(len(reward)):
            player_obs = observation[idx]
            player_x_pos = player_obs['left_team'][player_obs['active']][0]
            closest_def_pos = min(self.defensive_positions, key=lambda x: abs(x - player_x_pos))

            # Reward for reaching defensive positions and being ready to initiate counterattacks
            if closest_def_pos not in self.collected_positions.get(idx, []):
                if player_x_pos <= closest_def_pos:
                    components["defensive_position_reward"][idx] += 1.0
                    if idx not in self.collected_positions:
                        self.collected_positions[idx] = []
                    self.collected_positions[idx].append(closest_def_pos)

            # Calculating total reward with components
            reward[idx] += components["defensive_position_reward"][idx]

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
