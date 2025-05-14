import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on defensive strategies and team dynamics."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.distances_collected = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.distances_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['distances_collected'] = self.distances_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.distances_collected = from_pickle.get('distances_collected', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_positioning_reward": [0.0] * len(reward),
                      "stamina_management": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for index, o in enumerate(observation):
            player_pos = o['left_team'][o['active']]
            
            # Distance to own goal reward (encourage defensive positioning)
            own_goal_position = np.array([-1, 0])
            distance_to_goal = np.linalg.norm(player_pos - own_goal_position)
            distance_key = (index, 'distance_to_goal')
            
            if distance_key not in self.distances_collected:
                components["defensive_positioning_reward"][index] = 0.1 / (distance_to_goal + 0.1)
                self.distances_collected[distance_key] = distance_to_goal

            # Stamina management: reward for low tiredness
            tiredness = o['left_team_tired_factor'][o['active']]
            if tiredness < 0.1:
                components["stamina_management"][index] += 0.05
            else:
                components["stamina_management"][index] -= 0.05 * tiredness

            # Combine the rewards
            reward[index] += components["defensive_positioning_reward"][index] + components["stamina_management"][index]

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
