import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards advanced ball control and effective passing under pressure in tight game situations.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_accuracy_reward = 0.2
        self.control_under_pressure_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy(),
                      "pass_accuracy_reward": [0.0] * len(reward),
                      "control_under_pressure_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Check if the player is under high pressure (close opponents presence)
            player_pos = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]
            opponents = o['right_team'] if o['ball_owned_team'] == 0 else o['left_team']

            # Calculate pressure by distance to nearest opponent
            distances_to_opponents = np.linalg.norm(opponents - player_pos, axis=1)
            pressure = np.min(distances_to_opponents) < 0.1  # Arbitrary threshold for pressure

            if pressure:
                # Check if a successful pass has been made under pressure
                if any(o['sticky_actions'][7:10]):  # Assumption that indices 7, 8, 9 correspond to passing actions
                    components["pass_accuracy_reward"][rew_index] += self.pass_accuracy_reward
                    reward[rew_index] += components["pass_accuracy_reward"][rew_index]

                # Additional control reward when maintaining possession under pressure
                if o['ball_owned_player'] == o['active']:
                    components["control_under_pressure_reward"][rew_index] += self.control_under_pressure_reward
                    reward[rew_index] += components["control_under_pressure_reward"][rew_index]

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
