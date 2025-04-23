import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for defensive actions and strategic positioning, 
    enhancing defensive unit's capability to handle direct attacks and preparing counterattacks.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_reward = 0.2
        self.positioning_reward = 0.1
        self.defensive_positions = self.initialize_defensive_positions()
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            player_team = o['left_team' if o['active'] < len(o['left_team']) else 'right_team'][o['active']]
            opponent_goal = (1 if o['active'] < len(o['left_team']) else -1, 0)

            # Check distance to strategic defensive positions
            for position in self.defensive_positions:
                if np.linalg.norm(np.subtract(player_team, position)) < 0.1:
                    components["positioning_reward"][rew_index] = self.positioning_reward
                    reward[rew_index] += components["positioning_reward"][rew_index]

            # Add rewards for defensive actions if the player is near the opponent goal
            dist_to_goal = np.linalg.norm(np.subtract(player_team, opponent_goal))
            if dist_to_goal < 0.3 and o['ball_owned_team'] == (0 if o['active'] < len(o['left_team']) else 1):
                components["defensive_reward"][rew_index] = self.defensive_reward
                reward[rew_index] += components["defensive_reward"][rew_index]

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

    def initialize_defensive_positions(self):
        return [(0.5, 0.0), (-0.5, 0.0), (0.3, 0.2), (-0.3, -0.2)]
