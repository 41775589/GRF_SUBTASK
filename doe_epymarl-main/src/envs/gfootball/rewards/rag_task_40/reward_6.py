import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the defensive unit's capabilities to handle direct attacks.
    It encourages defensive players to improve confrontational defense and strategic positioning for counterattacks."""
    def __init__(self, env):
        super().__init__(env)
        self._dist_threshold = 0.1
        self._counter_attack_bonus = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_actions_taken = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_actions_taken = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._defensive_actions_taken
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._defensive_actions_taken = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 1:  # if the opposition has the ball
                opponents_dist = np.min(np.linalg.norm(o['right_team'] - o['ball'], axis=1))
                if opponents_dist < self._dist_threshold:
                    if rew_index not in self._defensive_actions_taken:
                        # Reward players for taking defensive actions when opponents are near
                        components["defensive_bonus"][rew_index] = self._counter_attack_bonus
                        reward[rew_index] += components["defensive_bonus"][rew_index]
                        self._defensive_actions_taken[rew_index] = True

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
