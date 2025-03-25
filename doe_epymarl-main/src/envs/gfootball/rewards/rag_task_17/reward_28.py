import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a custom reward for mastering wide midfield roles. It focuses on
    the ability to perform high passes while maintaining good field positioning.
    Minimizing opponent's control and assisting in stretching the opposition's defense.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_performance_memory = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_performance_memory = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        pickle_data = self.env.get_state(to_pickle)
        pickle_data['pass_performance_memory'] = self.pass_performance_memory
        return pickle_data

    def set_state(self, state):
        state_data = self.env.set_state(state)
        self.pass_performance_memory = state_data.get('pass_performance_memory', {})
        return state_data

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'passing_reward': [0.0] * len(reward),
                      'positioning_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            obs = observation[idx]
            components['base_score_reward'][idx] = reward[idx]

            # Reward for successful high passes
            if obs['sticky_actions'][9] == 1:  # high_pass
                high_pass_pos = np.array([obs['ball'][0], obs['ball'][1]])
                distance_to_goal = np.abs(high_pass_pos[0] - 1)  # Distance from right goal
                if distance_to_goal < 0.3:  # ball close to opponent's goal after pass
                    components['passing_reward'][idx] = 0.5 * reward[idx]

            # Positioning reward encouraging wide play
            player_pos = obs['left_team'][obs['active']] if obs['ball_owned_team'] == 0 else obs['right_team'][obs['active']]
            # Encourage players to stay wide on the pitch
            distance_from_center = np.abs(player_pos[1])  # Y coordinate abs value
            if distance_from_center > 0.3:  # if significantly lateral
                components['positioning_reward'][idx] = 0.2

            # Aggregate custom components to the reward
            reward[idx] += components['passing_reward'][idx]
            reward[idx] += components['positioning_reward'][idx]
        
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
            for i, action_state in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_state
        
        return observation, reward, done, info
