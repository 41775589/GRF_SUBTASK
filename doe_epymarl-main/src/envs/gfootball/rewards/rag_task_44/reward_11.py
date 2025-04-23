import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful stop-dribble maneuvers under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_dribble_counter = np.zeros(2, dtype=int)  # Assuming two agents
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.stop_dribble_counter = np.zeros(2, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['stop_dribble_counter'] = self.stop_dribble_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.stop_dribble_counter = from_pickle['stop_dribble_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "stop_dribble_reward": [0.0, 0.0]
        }
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            # Determine if the appropriate actions are observed
            if obs['sticky_actions'][9] == 1 and self.sticky_actions_counter[9] < 2:
                # Detect if the player switches from dribbling to staying put
                self.sticky_actions_counter[9] += 1
                if self.sticky_actions_counter[9] == 1:  # Dribbling initiated
                    continue
                elif self.sticky_actions_counter[9] == 2:  # Dribbling stopped
                    # Give reward only once after detecting stop-dribble sequence
                    self.stop_dribble_counter[idx] += 1
                    if self.stop_dribble_counter[idx] == 1:
                        reward[idx] += 1.0
                        components["stop_dribble_reward"][idx] = 1.0

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
