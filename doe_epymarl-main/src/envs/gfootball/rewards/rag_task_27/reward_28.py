import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized defensive ability and interception training reward to the base reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Parameters for interception rewards
        self._interception_reward = 0.5
        
        # Rewards specific to defensive positioning
        self._defensive_positions = [-0.9, -0.8, -0.7]  # Relative defensive focus regions
        self._defensive_position_rewards = [0.3, 0.2, 0.1]
        self._defensive_position_achieved = {key: False for key in range(len(self._defensive_positions))}
        
        # Rediscovery penalty
        self._reset_rediscovery_penalty = -0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_position_achieved = {key: False for key in self._defensive_position_achieved}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'interception_reward': [0.0] * len(reward),
                      'defensive_position_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Check interception
            if o['ball_owned_team'] == 1 and 'previous_ball_owned_team' in o and o['previous_ball_owned_team'] != 1:
                if o['active'] == o['ball_owned_player']:
                    components['interception_reward'][i] = self._interception_reward
                    reward[i] += components['interception_reward'][i]

            # Rewards based on historical defensive positional achievement
            player_x_pos = o['left_team'][o['active']][0]
            for idx, pos in enumerate(self._defensive_positions):
                if player_x_pos < pos and not self._defensive_position_achieved[idx]:
                    components['defensive_position_reward'][i] = self._defensive_position_rewards[idx]
                    reward[i] += components['defensive_position_reward'][i]
                    self._defensive_position_achieved[idx] = True
                elif player_x_pos > pos and self._defensive_position_achieved[idx]:
                    reward[i] += self._reset_rediscovery_penalty
                    self._defensive_position_achieved[idx] = False

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
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'defensive_position_achieved': self._defensive_position_achieved
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._defensive_position_achieved = from_pickle['CheckpointRewardWrapper']['defensive_position_achieved']
        return from_pickle
