import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds reward for handling abrupt stop and start actions, useful for defensive maneuvers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Counter for the sticky actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "stop_and_sprint_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Monitor sticky actions related to stopping and sprinting
            stopped = o['sticky_actions'][8]  # Sprint action
            moving_directions = o['sticky_actions'][:8]  # Movement directions
            moving = any(moving_directions)

            # Encourage stopping quickly after sprinting (stop-sprint mechanics)
            if stopped and not moving:
                if self.sticky_actions_counter[rew_index] < 5:  # reward for first 5 times stopped correctly
                    self.sticky_actions_counter[rew_index] += 1
                    components["stop_and_sprint_reward"][rew_index] = 0.2
                    reward[rew_index] += 0.2
            
            # Encourage moving quickly after stopping
            if not stopped and moving:
                if self.sticky_actions_counter[rew_index] < 5:  # reward for first 5 times moved correctly
                    self.sticky_actions_counter[rew_index] += 1
                    components["stop_and_sprint_reward"][rew_index] = 0.3
                    reward[rew_index] += 0.3

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
