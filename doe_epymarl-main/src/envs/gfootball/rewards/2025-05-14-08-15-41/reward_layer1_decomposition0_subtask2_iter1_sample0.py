import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering defensive tasks related to tackling and positioning strategically."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.incremental_position_reward = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.incremental_position_reward = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['PositionRewards'] = self.incremental_position_reward
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.incremental_position_reward = from_pickle.get('PositionRewards', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'defensive_positioning_reward': [0.0] * len(reward)}

        for idx, o in enumerate(observation):
            player_role = o.get('right_team_roles', [])[o['active']] if o['team'] == 'right' else o.get('left_team_roles', [])[o['active']]

            # Assuming defensive roles are coded as 1, 2, 3 (CB, LB, RB)
            if player_role in [1, 2, 3]:
                opposing_team_key = 'left_team' if o['team'] == 'right' else 'right_team'
                opposing_team = o[opposing_team_key]

                # Calculating defensive good positioning based on distance to all opponents
                defensive_reward = 0
                for opponent_pos in opposing_team:
                    distance = np.linalg.norm(np.array(o['position']) - np.array(opponent_pos))
                    if distance < 0.5:  # Encourage defending close to opponents
                        defensive_reward += 0.02

                # Update positional rewards based on continuous good defense
                idx_key = (o['team'], idx)
                self.incremental_position_reward[idx_key] = self.incremental_position_reward.get(idx_key, 0) + defensive_reward
                total_reward = self.incremental_position_reward[idx_key]
                
                components['defensive_positioning_reward'][idx] = total_reward
                reward[idx] += total_reward

        return reward, components

    def step(self, action):
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
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
