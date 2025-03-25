import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive skill and quick counter-attack training reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initiates counters for defensive and offensive transitions
        self.defensive_transitions = {}
        self.offensive_transitions = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_transitions = {}
        self.offensive_transitions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_transitions'] = self.defensive_transitions
        to_pickle['offensive_transitions'] = self.offensive_transitions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_transitions = from_pickle['defensive_transitions']
        self.offensive_transitions = from_pickle['offensive_transitions']
        return from_pickle

    def reward(self, reward):
        # Retrieve the observation from the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, (o, r) in enumerate(zip(observation, reward)):
            # Example of specific defensive and offensive mechanics
            # Detect ball loss near own goal
            ball_defensive_zone = o['ball'][0] < -0.5
            # Detect positioning for quick counter-attack near midfield
            ball_counterattack_ready = -0.2 < o['ball'][0] < 0.2

            if ball_defensive_zone and o['ball_owned_team'] != 0:
                if rew_index not in self.defensive_transitions:
                    components["defensive_reward"][rew_index] = 0.1
                    reward[rew_index] += components["defensive_reward"][rew_index]
                    self.defensive_transitions[rew_index] = True

            if ball_counterattack_ready and o['ball_owned_team'] == 0:
                if rew_index not in self.offensive_transitions:
                    components["counter_attack_reward"][rew_index] = 0.2
                    reward[rew_index] += components["counter_attack_reward"][rew_index]
                    self.offensive_transitions[rew_index] = True

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
