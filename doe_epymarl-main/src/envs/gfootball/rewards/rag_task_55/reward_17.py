import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents for effective defensive actions without fouling."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._max_tackle_without_foul = 5  # Maximum tackles without fouling
        self._positive_tackle_reward = 0.5
        self._negative_foul_penalty = -0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['is_penalty_for_foul'] = self._negative_foul_penalty
        to_pickle['max_tackles'] = self._max_tackle_without_foul
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._negative_foul_penalty = from_pickle.get('is_penalty_for_foul', -0.2)
        self._max_tackle_without_foul = from_pickle.get('max_tackles', 5)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackles_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, data in enumerate(zip(reward, observation)):
            rew, o = data
            if o['game_mode'] in {3, 6} and o['ball_owned_team'] == 1:  # fouls or penalties
                components["tackles_reward"][rew_index] = self._negative_foul_penalty
            elif o['game_mode'] == 0:  # normal game mode
                if 'slide' in o['sticky_actions'] or 'stand' in o['sticky_actions']:
                    if self.sticky_actions_counter[rew_index] < self._max_tackle_without_foul:
                        components["tackles_reward"][rew_index] = self._positive_tackle_reward
                        self.sticky_actions_counter[rew_index] += 1
            rew += components["tackles_reward"][rew_index]
            reward[rew_index] = rew

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info["component_{}".format(key)] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info["sticky_actions_{}".format(i)] = action
        return observation, reward, done, info
