import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that enhances defending strategies by introducing rewards for tackling, movement control, 
    and efficient passing in high-pressure situations.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.tackle_reward_coefficient = 0.3
        self.movement_control_coefficient = 0.2
        self.pressure_pass_coefficient = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = dict(tackle_reward_coefficient=self.tackle_reward_coefficient,
                                                    movement_control_coefficient=self.movement_control_coefficient,
                                                    pressure_pass_coefficient=self.pressure_pass_coefficient)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackle_reward_coefficient = from_pickle['tackle_reward_coefficient']
        self.movement_control_coefficient = from_pickle['movement_control_coefficient']
        self.pressure_pass_coefficient = from_pickle['pressure_pass_coefficient']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "movement_control_reward": [0.0] * len(reward),
                      "pressure_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for successful tackles - triggered if ball possession changes under pressure
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0 and reward[rew_index] == 0:
                components["tackle_reward"][rew_index] = self.tackle_reward_coefficient
                reward[rew_index] += components["tackle_reward"][rew_index]

            # Reward for movement control
            if 'left_team_active' in o and not o['left_team_active'][o['active']]:
                components["movement_control_reward"][rew_index] = self.movement_control_coefficient
                reward[rew_index] += components["movement_control_reward"][rew_index]

            # Reward for successful passes under pressure
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0 and
                'sticky_actions' in o and o['sticky_actions'][9] == 1):  # assuming index 9 is pass
                components["pressure_pass_reward"][rew_index] = self.pressure_pass_coefficient
                reward[rew_index] += components["pressure_pass_reward"][rew_index]
        
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
