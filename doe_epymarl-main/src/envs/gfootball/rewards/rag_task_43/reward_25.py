import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive strategy and counterattack transition reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positioning_reward = 0.01  # Reward for maintaining positioning and distance appropriately
        self.defensive_rewards = {}
        self.active_positions_tracker = {}

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_rewards = {}
        self.active_positions_tracker = {}
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['DefensiveRewards'] = self.defensive_rewards
        to_pickle['ActivePositionsTracker'] = self.active_positions_tracker
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_rewards = from_pickle['DefensiveRewards']
        self.active_positions_tracker = from_pickle['ActivePositionsTracker']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            ball_pos = o['ball'][:2]  # Only consider x, y

            # Calculate distances to ball
            distance = np.linalg.norm(player_pos - ball_pos)

            # Encourage players to maintain an optimal distance from the ball
            optimal_distance = 0.1  # optimal distance to the ball in normalized coordinates
            if distance < optimal_distance:
                components["defensive_reward"][rew_index] = self.positioning_reward
                reward[rew_index] += components["defensive_reward"][rew_index]

            # Track positions and check if maintaining formation
            if rew_index not in self.active_positions_tracker:
                self.active_positions_tracker[rew_index] = [player_pos]
            else:
                self.active_positions_tracker[rew_index].append(player_pos)
            if len(self.active_positions_tracker[rew_index]) > 5:  # Check last 5 positions
                prev_positions = self.active_positions_tracker[rew_index][-5:]
                variance = np.var(prev_positions, axis=0)
                if np.any(variance < 0.01):  # Low variance indicates good positioning
                    components["defensive_reward"][rew_index] += 0.02
                    reward[rew_index] += 0.02

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions info for each step
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
