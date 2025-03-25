import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances teamwork in defensive scenarios near the penalty area."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._penalty_area_threshold_x = 0.6  # approximated penalty area threshold
        self._penalty_area_threshold_y = 0.2  # approximated penalty area height
        self._reward_for_defensive_effort = 0.15
        self._reward_for_partnership = 0.05

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
        components = {
            "base_score_reward": reward.copy(),
            "defensive_effort_reward": [0.0] * len(reward),
            "partnership_reward": [0.0] * len(reward),
            "no_reward_condition": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        # Iterate through the observations for both agents
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check defensive positioning in the penalty area
            player_x, player_y = o['left_team'][o['active']][:2]
            if abs(player_x) > self._penalty_area_threshold_x and abs(player_y) < self._penalty_area_threshold_y:
                # Defensive effort
                components["defensive_effort_reward"][rew_index] = self._reward_for_defensive_effort
                reward[rew_index] += components["defensive_effort_reward"][rew_index]
                
                # Check synergy with another defender
                for other_index in range(len(o['left_team'])):
                    if other_index != o['active']:
                        teammate_x, teammate_y = o['left_team'][other_index][:2]
                        if (abs(teammate_x) > self._penalty_area_threshold_x and abs(teammate_y) < self._penalty_area_threshold_y):
                            components["partnership_reward"][rew_index] = self._reward_for_partnership
                            reward[rew_index] += components["partnership_reward"][rew_index]
                            break

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
