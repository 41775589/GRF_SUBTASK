import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive maneuvers and tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_tackles = np.zeros(2)  # Number of tackles per player
        self.last_slides = np.zeros(2)  # Number of slide tackles per player

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_tackles = np.zeros(2)
        self.last_slides = np.zeros(2)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['tackles'] = self.last_tackles
        to_pickle['slides'] = self.last_slides
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_tackles = from_pickle['tackles']
        self.last_slides = from_pickle['slides']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": np.zeros(2),
                      "slide_reward": np.zeros(2)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for agent_index in range(len(reward)):
            o = observation[agent_index]

            # Reward for successful tackles
            current_tackles = o['sticky_actions'][10]  # Assuming index 10 for tackle action
            if current_tackles > self.last_tackles[agent_index]:
                components["tackle_reward"][agent_index] = 0.2  # Reward for each tackle
                self.last_tackles[agent_index] = current_tackles

            # Reward for sliding tackles
            current_slides = o['sticky_actions'][11]  # Assuming index 11 for slide tackle action
            if current_slides > self.last_slides[agent_index]:
                components["slide_reward"][agent_index] = 0.5  # Higher reward for riskier slide tackle
                self.last_slides[agent_index] = current_slides

            # Update the reward for this agent
            reward[agent_index] += (components["tackle_reward"][agent_index] +
                                    components["slide_reward"][agent_index])

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
