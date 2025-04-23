import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on rewarding short pass, long pass, and dribbling under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._pass_progress_reward = 0.1
        self._dribble_progress_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward}
        
        components = {
            'base_score_reward': reward.copy(),
            'pass_progress_reward': [0.0] * len(reward),
            'dribble_progress_reward': [0.0] * len(reward)
        }

        for rew_index, o in enumerate(observation):
            if o['ball_owned_team'] == 0: # Assuming 0 is the index for the team we are rewarding
                active_player = o['active']
                sticky_actions = o['sticky_actions']

                # Reward for passing mechanisms: check for short and long pass sticky actions
                if sticky_actions[0] == 1 or sticky_actions[1] == 1: # Assuming indices 0 and 1 stand for short and long passes respectively
                    components['pass_progress_reward'][rew_index] = self._pass_progress_reward
                    reward[rew_index] += self._pass_progress_reward

                # Reward for dribbling under pressure: check for dribble sticky action
                if sticky_actions[9] == 1: # Assuming index 9 stands for dribble
                    components['dribble_progress_reward'][rew_index] = self._dribble_progress_reward
                    reward[rew_index] += self._dribble_progress_reward

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
