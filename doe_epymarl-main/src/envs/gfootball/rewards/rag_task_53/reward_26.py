import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for maintaining ball control and strategic positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define reward modifiers
        self.ball_control_reward = 0.1
        self.positioning_reward = 0.2
        self._has_ball_control = False

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._has_ball_control = False
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle = self.env.get_state(to_pickle)
        to_pickle['has_ball_control'] = self._has_ball_control
        return to_pickle

    def set_state(self, state):
        state = self.env.set_state(state)
        self._has_ball_control = state.get('has_ball_control', False)

    def reward(self, reward):
        observation = self.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "ball_control_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            
            # Ball control reward
            if o['ball_owned_team'] == 1:
                self._has_ball_control = True

            if self._has_ball_control and o['ball_owned_player'] == o['active']:
                components['ball_control_reward'][rew_index] = self.ball_control_reward
                reward[rew_index] += 0.5 * components['ball_control_reward'][rew_index]
            
            # Positioning reward: encourage exploring open space
            if o['ball_owned_team'] == 1:
                # Calculate distances from all opponents to measure how open the space is
                distances = np.sqrt(np.sum((o['right_team'] - o['ball'][:2])**2, axis=1))
                min_distance = np.min(distances)
                if min_distance > 0.2:
                    components['positioning_reward'][rew_index] = self.positioning_reward
                    reward[rew_index] += 0.5 * components['positioning_reward'][rew_index]

            self._has_ball_control = False

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
