import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that promotes energy conservation via efficient use of Stop-Sprint and Stop-Moving actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_stop_counter = np.zeros(2, dtype=int)  # Tracks sprint stops for both players
        self.movement_stop_counter = np.zeros(2, dtype=int)  # Tracks movement stops for both players
        self.rewards = {"sprint_stop_reward": 0.05, "movement_stop_reward": 0.05}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.sprint_stop_counter.fill(0)
        self.movement_stop_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sprint_stop_counter'] = self.sprint_stop_counter.copy()
        to_pickle['movement_stop_counter'] = self.movement_stop_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_stop_counter = from_pickle['sprint_stop_counter']
        self.movement_stop_counter = from_pickle['movement_stop_counter']
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "sprint_stop_reward": [0.0, 0.0],
                      "movement_stop_reward": [0.0, 0.0]}
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components

        for i, rew in enumerate(reward):
            if 'sticky_actions' in observation[i]:
                current_actions = observation[i]['sticky_actions']

                # Sprint action indices may vary, here assuming it's at index 8
                # Sprint stop is rewarded only if the sprint action was active and is now stopped
                if self.sticky_actions_counter[8] == 1 and current_actions[8] == 0:
                    self.sprint_stop_counter[i] += 1
                    reward[i] += self.rewards['sprint_stop_reward']
                    components["sprint_stop_reward"][i] = self.rewards['sprint_stop_reward']
                
                # Assumption: indices 0-7 are moving actions
                # A movement stop is rewarded when all movement actions go to 0 from a previous non-zero state
                if sum(self.sticky_actions_counter[:8]) > 0 and sum(current_actions[:8]) == 0:
                    self.movement_stop_counter[i] += 1
                    reward[i] += self.rewards['movement_stop_reward']
                    components["movement_stop_reward"][i] = self.rewards['movement_stop_reward']

            self.sticky_actions_counter = current_actions.copy()

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
