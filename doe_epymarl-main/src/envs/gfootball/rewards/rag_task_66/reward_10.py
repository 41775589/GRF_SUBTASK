import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward based on successful short passes under pressure for training ball retention and distribution."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._successful_passes = 0
        self._pass_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._successful_passes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['successful_passes'] = self._successful_passes
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._successful_passes = from_pickle.get('successful_passes', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['game_mode'] == 0 and o['ball_owned_team'] == 0:  # Normal play and ball owned by agent's team
                if 'action' in o.keys() and o['action'] == 'short_pass' and self._previous_ball_owner != o['active']:
                    components["pass_reward"][rew_index] = self._pass_reward
                    self._successful_passes += 1
                reward[rew_index] += components["pass_reward"][rew_index]

        self._previous_ball_owner = observation[0]['ball_owned_player'] if len(observation) > 0 else -1

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
