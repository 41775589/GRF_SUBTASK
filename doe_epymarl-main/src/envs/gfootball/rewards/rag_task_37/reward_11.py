import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to encourage advanced ball control and passing under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_passes = 0
        self._pass_bonus = 0.2
        self._control_bonus = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_passes = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'num_passes': self._num_passes}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._num_passes = from_pickle['CheckpointRewardWrapper']['num_passes']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_bonus": [0.0] * len(reward),
                      "control_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'sticky_actions' not in o:
                continue

            # Process for ball control and passing
            # Check if the player has the ball
            has_ball = (o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active'])

            # Sticky Actions indices for passing: 1 (Short Pass), 2 (High Pass), 5 (Long Pass)
            if has_ball and (o['sticky_actions'][1] or o['sticky_actions'][2] or o['sticky_actions'][5]):
                components['pass_bonus'][rew_index] = self._pass_bonus
                reward[rew_index] += components['pass_bonus'][rew_index]
                self._num_passes += 1

            # Check for tight game situations (opponents close)
            opponent_distances = np.sqrt(np.sum((o['right_team'] - o['ball'][0:2]) ** 2, axis=1))
            # Assuming a tight situation if any opponent is within a 0.1 radius
            if np.any(opponent_distances < 0.1) and has_ball:
                components['control_bonus'][rew_index] = self._control_bonus
                reward[rew_index] += components['control_bonus'][rew_index]

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
