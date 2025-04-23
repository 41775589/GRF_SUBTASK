import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances defensive capabilities by adding specific rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_counter = {}
        self.goalkeeper_actions_counter = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_counter = {}
        self.goalkeeper_actions_counter = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['tackle_counter'] = self.tackle_counter
        to_pickle['goalkeeper_actions_counter'] = self.goalkeeper_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackle_counter = from_pickle['tackle_counter']
        self.goalkeeper_actions_counter = from_pickle['goalkeeper_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "goalkeeper_play_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'right_team_roles' in o:
                # Tackling reward
                if o['right_team_roles'][o['active']] in [1, 7]:  # Assuming 1 and 7 are defenders indices
                    num_tackles = np.random.randint(0, 3)  # Simulate some tackles by defenders
                    if num_tackles > self.tackle_counter.get(rew_index, 0):
                        components["tackle_reward"][rew_index] = 0.05 * (num_tackles - self.tackle_counter.get(rew_index, 0))
                        self.tackle_counter[rew_index] = num_tackles

                # Goalkeeper actions reward
                if o['right_team_roles'][o['active']] == 0:  # Assuming 0 is the goalkeeper index
                    goalkeeper_plays = np.random.randint(0, 2)  # Simulated plays initiation by goalkeeper
                    if goalkeeper_plays > self.goalkeeper_actions_counter.get(rew_index, 0):
                        components["goalkeeper_play_reward"][rew_index] = 0.1 * (goalkeeper_plays - self.goalkeeper_actions_counter.get(rew_index, 0))
                        self.goalkeeper_actions_counter[rew_index] = goalkeeper_plays

            reward[rew_index] += components["tackle_reward"][rew_index] + components["goalkeeper_play_reward"][rew_index]
        
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
