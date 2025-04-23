import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on maintaining ball control under pressure and effective space utilization."""

    def __init__(self, env):
        super().__init__(env)
        self.pressure_coefficient = 0.05
        self.space_utilization_coefficient = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'pressure_coefficient': self.pressure_coefficient, 'space_utilization_coefficient': self.space_utilization_coefficient}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.pressure_coefficient = state_data['pressure_coefficient']
        self.space_utilization_coefficient = state_data['space_utilization_coefficient']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pressure_reward": [0.0] * len(reward),
                      "space_utilization_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, obs in enumerate(observation):
            # Maintaining Ball Control under Pressure
            if obs['ball_owned_team'] == 0 and any(obs['left_team_active']):
                ball_owner_pos = obs['left_team'][obs['ball_owned_player']]
                pressure = sum([np.linalg.norm(ball_owner_pos - pos) < 0.1 for pos in obs['right_team']])
                components["pressure_reward"][index] += self.pressure_coefficient * pressure

            # Effective Space Utilization 
            if obs['ball_owned_team'] == 0:
                # Calculate distance towards opponent's goal, normalized over field length
                distance_to_goal = 1 - obs['ball'][0]  # field length assumed from -1 to 1
                components["space_utilization_reward"][index] += self.space_utilization_coefficient * distance_to_goal

            # Sum all rewards including the base and additional rewards
            reward[index] += components["pressure_reward"][index] + components["space_utilization_reward"][index]
             
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
