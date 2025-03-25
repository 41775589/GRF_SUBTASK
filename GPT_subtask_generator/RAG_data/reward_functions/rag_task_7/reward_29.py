import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering defensive maneuvers, specifically precision in sliding tackles under high-pressure situations."""

    def __init__(self, env):
        super().__init__(env)
        self._num_samples = 0
        self._successful_tackles = 0
        self._tackle_initiated = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self._num_samples = 0
        self._successful_tackles = 0
        self._tackle_initiated = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['num_samples'] = self._num_samples
        to_pickle['successful_tackles'] = self._successful_tackles
        to_pickle['tackle_initiated'] = self._tackle_initiated
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._num_samples = from_pickle.get('num_samples', 0)
        self._successful_tackles = from_pickle.get('successful_tackles', 0)
        self._tackle_initiated = from_pickle.get('tackle_initiated', False)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_success_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if 'ball_owned_player' in o and o['ball_owned_player'] != o['active']:
                self._tackle_initiated = True
            
            if self._tackle_initiated and o['game_mode'] == 6:  # Indicates a slide tackle event
                if o['ball_owned_team'] == 0:  # Ball owned by the environment's team
                    components["tackle_success_reward"][rew_index] = 1.0  # Reward for successful tackle
                    self._successful_tackles += 1
            
            reward[rew_index] += components["tackle_success_reward"][rew_index]

        self._num_samples += 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        info["tackle_efficiency"] = (self._successful_tackles / self._num_samples if self._num_samples > 0 else 0)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
