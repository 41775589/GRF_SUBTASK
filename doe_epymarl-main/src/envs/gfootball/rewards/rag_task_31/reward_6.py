import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defense-oriented reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackles_counter = {}
        self._slide_tackles_counter = {}
        self._num_checkpoints = 5
        self._defense_checkpoint_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackles_counter.clear()
        self._slide_tackles_counter.clear()
        return super().reset()

    def get_state(self, to_pickle):
        to_pickle['tackles_counter'] = self._tackles_counter
        to_pickle['slide_tackles_counter'] = self._slide_tackles_counter
        return super().get_state(to_pickle)

    def set_state(self, from_pickle):
        state = super().set_state(from_pickle)
        self._tackles_counter = from_pickle['tackles_counter']
        self._slide_tackles_counter = from_pickle['slide_tackles_counter']
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_checkpoint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if reward[rew_index] == -1:  # The opposite team scores
                # Penalize the reward to encourage more defense
                reward[rew_index] -= 0.5
                continue

            # Check defensive actions like tackles and sliding
            if 'sticky_actions' in o:
                # Tackle actions can be identified, assuming index 5 and 6
                tackle_action = o['sticky_actions'][5]
                slide_action = o['sticky_actions'][6]

                # Encourage tackles
                if tackle_action:
                    curr_tackles = self._tackles_counter.get(rew_index, 0)
                    if curr_tackles < self._num_checkpoints:
                        components['defense_checkpoint_reward'][rew_index] += self._defense_checkpoint_reward
                        reward[rew_index] += components['defense_checkpoint_reward'][rew_index]
                        self._tackles_counter[rew_index] = curr_tackles + 1

                # Encourage slide tackles
                if slide_action:
                    curr_slides = self._slide_tackles_counter.get(rew_index, 0)
                    if curr_slides < self._num_checkpoints:
                        components['defense_checkpoint_reward'][rew_index] += self._defense_checkpoint_reward * 1.5
                        reward[rew_index] += components['defense_checkpoint_reward'][rew_index]
                        self._slide_tackles_counter[rew_index] = curr_slides + 1

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
