import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for additional rewards.
        self.tackle_reward = 0.2
        self.pass_under_pressure_reward = 0.3
        self.positioning_reward = 0.1
        self._pressured_threshold = 0.2  # Threshold to determine if a passing occurred under pressure

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
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "pass_under_pressure_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            if 'ball_owned_player' in o and o['ball_owned_player'] == o['active']:
                # Tackling reward: When ball is recovered by player while opposing player is nearby.
                proximity_to_opponents = np.sqrt(np.sum((o['left_team'][o['active']] - o['right_team'])**2, axis=1))
                close_opponents = np.sum(proximity_to_opponents < 0.1)
                if close_opponents > 0:
                    reward[idx] += self.tackle_reward
                    components['tackle_reward'][idx] = self.tackle_reward

                # Passing under pressure: Successfully passes while opponents are nearby.
                if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Assuming index 9 is pass action
                    if close_opponents > 0:
                        reward[idx] += self.pass_under_pressure_reward
                        components['pass_under_pressure_reward'][idx] = self.pass_under_pressure_reward
            
            # Positioning reward: Encourage maintaining strategic positions for defense.
            # Assume active player's position close to goal is strategic positioning.
            defensive_line_x = -0.75  # Strategic defensive x line
            if o['right_team'][o['active']][0] <= defensive_line_x:
                reward[idx] += self.positioning_reward
                components['positioning_reward'][idx] = self.positioning_reward

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
        return observation, reward, done, info
