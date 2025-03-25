import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes immediate ball possession recovery and quick counter-attacks.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.recovery_reward = 0.5
        self.counter_attack_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        current_observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": list(reward),
            "recovery_reward": [0.0, 0.0],
            "counter_attack_reward": [0.0, 0.0]
        }

        if current_observation is None:
            return reward, components

        for agent_idx, observation in enumerate(current_observation):
            if observation['ball_owned_team'] == 0 and observation['ball_owned_player'] == observation['active']:
                # Ball possession recovery reward
                if previous_obs and previous_obs[agent_idx]['ball_owned_team'] != 0:
                    reward[agent_idx] += self.recovery_reward
                    components["recovery_reward"][agent_idx] += self.recovery_reward

                # Aggressive forward movement reward (counter-attack)
                if observation['ball'][0] > 0 and np.any(observation['right_team_direction'][:, 0] > 0):
                    reward[agent_idx] += self.counter_attack_reward
                    components["counter_attack_reward"][agent_idx] += self.counter_attack_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
