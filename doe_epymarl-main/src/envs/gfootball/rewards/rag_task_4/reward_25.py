import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds an advanced dribbling and sprinting reward."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_progress = {}
        self.sprint_usage = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_progress = {}
        self.sprint_usage = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state.update({
            'dribble_progress': self.dribble_progress,
            'sprint_usage': self.sprint_usage
        })
        return state

    def set_state(self, state):
        self.dribble_progress = state['dribble_progress']
        self.sprint_usage = state['sprint_usage']
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'dribble_reward': [0.0, 0.0],
                      'sprint_reward': [0.0, 0.0]}
        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if obs['active'] == -1:
                continue

            # Reward for using dribble in tight situations
            if obs['sticky_actions'][9] == 1:  # Action 9 is 'dribble'
                self.dribble_progress[i] = self.dribble_progress.get(i, 0) + 1
            else:
                self.dribble_progress[i] = 0

            if self.dribble_progress[i] > 3:
                components['dribble_reward'][i] = 0.02
                reward[i] += components['dribble_reward'][i]

            # Reward for using sprint creatively
            if obs['sticky_actions'][8] == 1:  # Action 8 is 'sprint'
                self.sprint_usage[i] = self.sprint_usage.get(i, 0) + 1
            else:
                self.sprint_usage[i] = 0

            if self.sprint_usage[i] >= 5 and self.dribble_progress[i] > 3: 
                components['sprint_reward'][i] = 0.05
                reward[i] += components['sprint_reward'][i]

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
