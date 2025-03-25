import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic offensive maneuver reward based on ball control and quick attack adaptation."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goals_scored = 0
        self.offensive_phase_score = 0.1  # Reward for maintaining ball in opponent half
        self.attack_completion_score = 0.5  # Extra reward for scoring a goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goals_scored = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['goals_scored'] = self.goals_scored
        return state

    def set_state(self, state):
        self.goals_scored = state.get('goals_scored', 0)
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_phase_reward": [0.0] * len(reward),
                      "attack_completion_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            # Check if the active player is in the opponents half
            if o['ball'][0] > 0:  # Ball is in opponents half
                components["offensive_phase_reward"][idx] = self.offensive_phase_score
                reward[idx] += components["offensive_phase_reward"][idx]

            # Extra reward for scoring
            if o['score'][0] > self.goals_scored:  # Assumes that team of index 0 is the one we're controlling
                self.goals_scored = o['score'][0]
                reward[idx] += self.attack_completion_score
                components["attack_completion_reward"][idx] = self.attack_completion_score

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
