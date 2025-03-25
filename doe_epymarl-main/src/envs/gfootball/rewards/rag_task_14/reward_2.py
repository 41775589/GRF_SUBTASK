import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on the 'sweeper' role in the defense."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearances = 0
        self.tackles = 0
        self.coverages = 0
        self.clearance_reward = 0.5
        self.tackle_reward = 0.7
        self.coverage_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearances = 0
        self.tackles = 0
        self.coverages = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['SweeperRewards'] = (self.clearances, self.tackles, self.coverages)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.clearances, self.tackles, self.coverages = from_pickle['SweeperRewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'clearance_reward': [0.0] * len(reward),
                      'tackle_reward': [0.0] * len(reward),
                      'coverage_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        active_players = self.env.unwrapped.get_active_players()

        for i, player in enumerate(active_players):
            if player['role'] == 8:  # Assuming 8 is sweeper role
                # Check for clearances
                if self._cleared(observation[i]):
                    self.clearances += 1
                    reward[i] += self.clearance_reward
                    components['clearance_reward'][i] = self.clearance_reward

                # Check for tackles
                if self._tackled(observation[i]):
                    self.tackles += 1
                    reward[i] += self.tackle_reward
                    components['tackle_reward'][i] = self.tackle_reward

                # Check for coverage
                if self._covered(observation[i]):
                    self.coverages += 1
                    reward[i] += self.coverage_reward
                    components['coverage_reward'][i] = self.coverage_reward

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

    def _cleared(self, observation):
        # Example logic for football environment clearance detection
        return observation['ball_owned_team'] == 0 and np.linalg.norm(observation['ball_direction']) > 1

    def _tackled(self, observation):
        # Example logic for football environment tackle detection
        return np.random.choice([True, False])  # Simplified example

    def _covered(self, observation):
        # Example logic for football environment coverage detection
        return np.linalg.norm(observation['right_team'][observation['active']]) > 1
