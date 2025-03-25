import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for a hybrid midfielder/defender."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._pass_bonus = 0.1
        self._dribble_bonus = 0.2
        self._sprint_bonus = 0.05
        self._stop_sprint_penalty = -0.05
        self._tracked_actions = {
            'high_pass': 8,
            'long_pass': 9,
            'dribble': 1,
            'sprint': 8,
            'stop_sprint': 9,
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "action_bonus": [0.0] * len(reward)}
        if not observation:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            action_ids = o.get('sticky_actions', [])

            # Reward for specific actions reflecting the skills of a midfielder/defender
            if any(action_ids):  # Checking if any action is taken
                components["action_bonus"][rew_index] = sum([
                    self._pass_bonus if action_ids[self._tracked_actions['high_pass']] else 0,
                    self._pass_bonus if action_ids[self._tracked_actions['long_pass']] else 0,
                    self._dribble_bonus if action_ids[self._tracked_actions['dribble']] else 0,
                    self._sprint_bonus if action_ids[self._tracked_actions['sprint']] else 0,
                    self._stop_sprint_penalty if action_ids[self._tracked_actions['stop_sprint']] else 0,
                ])
                # Applying reward modification
                reward[rew_index] += components["action_bonus"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Resetting sticky actions after their effects were accounted
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
