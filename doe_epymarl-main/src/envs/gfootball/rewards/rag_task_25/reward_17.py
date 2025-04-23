import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on dribbling and sprinting proficiency."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the sticky actions counter on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the CheckpointRewardWrapper to save its state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the CheckpointRewardWrapper based on loaded state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Compute the reward based on dribbling and using sprint."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        dribble_weight = 0.1
        sprint_weight = 0.2

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            dribble_active = o['sticky_actions'][9] == 1  # Dribble action index
            sprint_active = o['sticky_actions'][8] == 1  # Sprint action index

            if dribble_active:
                components['dribble_sprint_reward'][rew_index] += dribble_weight
            if sprint_active:
                components['dribble_sprint_reward'][rew_index] += sprint_weight

            reward[rew_index] += components['dribble_sprint_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Steps the environment by one timestep."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
