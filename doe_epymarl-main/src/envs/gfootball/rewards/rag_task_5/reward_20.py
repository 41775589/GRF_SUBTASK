import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specialized defensive and counter-attack training rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positive_defensive_actions = 0
        self.ball_recovery_counter = 0
        self.counter_attack_efficiency = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.positive_defensive_actions = 0
        self.ball_recovery_counter = 0
        self.counter_attack_efficiency = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['positive_defensive_actions'] = self.positive_defensive_actions
        to_pickle['ball_recovery_counter'] = self.ball_recovery_counter
        to_pickle['counter_attack_efficiency'] = self.counter_attack_efficiency
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.positive_defensive_actions = from_pickle.get('positive_defensive_actions', 0)
        self.ball_recovery_counter = from_pickle.get('ball_recovery_counter', 0)
        self.counter_attack_efficiency = from_pickle.get('counter_attack_efficiency', 0)
        return from_pickle

    def reward(self, reward):
        # Incorporate the custom reward modifications
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_action_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Increase reward for successful takeways in the defensive third
            if o['ball_owned_team'] == 1 and o['ball'][0] < -0.5:
                self.ball_recovery_counter += 1
                components['defensive_action_reward'][i] = self.ball_recovery_counter * 0.05
                reward[i] += components['defensive_action_reward'][i]
            
            # Increase reward for quick transitions to counter-attack
            if o['ball_owned_team'] == 1 and o['ball'][0] > 0.5:
                self.counter_attack_efficiency += 1
                components['counter_attack_reward'][i] = self.counter_attack_efficiency * 0.1
                reward[i] += components['counter_attack_reward'][i]

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
