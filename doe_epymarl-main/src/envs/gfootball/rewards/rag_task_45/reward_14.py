import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards based on abrupt stopping and sprint actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # 0-indexed

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Resetting the sticky action counter
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Rewards agents for effectively stopping abruptly and then sprinting."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            current_obs = observation[rew_index]
            # Identifying abrupt stops and quick sprints (based on sticky actions)
            is_sprinting = current_obs['sticky_actions'][8]  # action_sprint index
            action_movement = np.array(current_obs['sticky_actions'][0:8])  # all movement actions

            was_sprinting = self.sticky_actions_counter[8]
            was_moving = np.any(self.sticky_actions_counter[0:8])

            # Define the reward for stopping abruptly and starting to sprint
            if was_moving and not np.any(action_movement) and is_sprinting:
                components["stop_sprint_reward"][rew_index] = 0.2  # Sprinting after stop gives a reward
                reward[rew_index] += components["stop_sprint_reward"][rew_index]
            
            # Update the historical counters for sticky actions
            self.sticky_actions_counter.fill(0)
            action_indices = np.where(current_obs['sticky_actions'])[0]
            self.sticky_actions_counter[action_indices] = 1

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
