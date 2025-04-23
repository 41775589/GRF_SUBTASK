import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for shooting accurately from central field positions."""

    def __init__(self, env):
        super().__init__(env)
        self._accuracy_threshold = 0.15
        self._power_threshold = 0.8
        self._accuracy_reward = 1.0
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
                      "accuracy_power_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = np.array(o['ball'][:2])  # only consider x, y
            
            # Check if near central field
            if abs(ball_pos[0]) < self._accuracy_threshold and o['ball_owned_team'] == 0:
                # Check shot power and accuracy
                ball_speed = np.linalg.norm(o['ball_direction'][:2])
                if ball_speed > self._power_threshold:
                    components["accuracy_power_reward"][rew_index] = self._accuracy_reward
                    reward[rew_index] += self._accuracy_reward

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
