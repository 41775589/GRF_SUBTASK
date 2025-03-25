import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward component for sprinting effectively in defensive positions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.max_sprint_intervals = 10
        self.sprint_reward = 0.05
        self.defensive_zones = [
            (0, 0.1), (0.1, 0.2), (0.2, 0.3),
            (0.3, 0.4), (0.4, 0.5), (0.5, 0.6),
            (0.6, 0.7), (0.7, 0.8), (0.8, 0.9),
            (0.9, 1.0)
        ]
        self._sprints_collected = {zone: 0 for zone in self.defensive_zones}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._sprints_collected = {zone: 0 for zone in self.defensive_zones}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._sprints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._sprints_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "sprint_bonus": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            if 'sticky_actions' in o:
                sprint_action_active = o['sticky_actions'][8]
                player_x_position = o['right_team'][o['active']][0]
                # Calculate effective sprint distances and adding rewards for sprints in specific defensive zones
                for zone, max_sprint in self._sprints_collected.items():
                    if zone[0] <= player_x_position < zone[1] and max_sprint < self.max_sprint_intervals:
                        if sprint_action_active == 1:
                            bonus = self.sprint_reward
                            components["sprint_bonus"][rew_index] += bonus
                            self._sprints_collected[zone] += 1
                            reward[rew_index] += bonus

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
