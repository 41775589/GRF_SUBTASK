import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward based on successful standing tackles, focusing on precision and control during defensive situations."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.tackle_success_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.5

    def reset(self):
        """Resets the environment and tackle counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_success_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Saves the current state, including custom tackle count states."""
        to_pickle['tackle_success_counter'] = self.tackle_success_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state, including custom tackle count states."""
        from_pickle = self.env.set_state(state)
        self.tackle_success_counter = from_pickle.get('tackle_success_counter', 0)
        return from_pickle

    def reward(self, reward):
        """Adjusts the rewards by adding bonuses for successful tackles during gameplay."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]
            
            # Reward players for successful tackles based on game mode
            if o['game_mode'] in [0, 3]:  # Normal and Set-Piece (e.g., Free Kicks)
                if 'ball_owned_team' in o and o['ball_owned_team'] == o['active'] - 1:
                    # Successfully tackled and gained possession
                    reward[rew_index] += self.tackle_reward
                    components["tackle_reward"][rew_index] = self.tackle_reward
                    self.tackle_success_counter += 1

        return reward, components

    def step(self, action):
        """Executes a step in the environment with adjusted components and rewards."""
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
