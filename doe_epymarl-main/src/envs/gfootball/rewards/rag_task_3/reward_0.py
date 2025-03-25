import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward for practicing shots with different pressure scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_attempt_reward = 1.0
        self.goal_reward = 2.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_attempt_reward": [0.0] * len(reward),
                      "goal_score_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'game_mode' in o:
                # Check if a goal was scored
                if o['game_mode'] == 6:   # GameMode.PENALTY = 6
                    reward[rew_index] += self.goal_reward
                    components["goal_score_reward"][rew_index] = self.goal_reward
                # Check if the current sticky action is a shot
                if o['sticky_actions'][9] == 1:  # Index 9 corresponds to 'Shot' action
                    reward[rew_index] += self.shot_attempt_reward
                    components["shot_attempt_reward"][rew_index] = self.shot_attempt_reward

                    # Updating sticky_actions_counter to record the shot action execution
                    self.sticky_actions_counter[9] += 1
        
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
