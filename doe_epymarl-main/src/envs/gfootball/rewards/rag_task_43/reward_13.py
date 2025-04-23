import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive positional awareness and counterattack reward."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackling_reward_factor = 0.1
        self._counterattack_reward_factor = 0.2

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
        components = {'base_score_reward': reward.copy(),
                      'tackling_reward': [0.0] * len(reward),
                      'counterattack_reward': [0.0] * len(reward)}
        if observation is None:
            return reward, components
        
        for i, obs in enumerate(observation):
            # Reward tackling: close distance to opponent with the ball
            if obs['ball_owned_team'] == 1:  # ball owned by opponent
                distance_to_ball = np.linalg.norm(obs['left_team'][obs['active']] - obs['ball'][:2])
                if distance_to_ball < 0.1:  # arbitrary threshold for close distance
                    components['tackling_reward'][i] = self._tackling_reward_factor

            # Reward counterattack: rapid movement towards opponent's goal when ball is gained
            if obs['ball_owned_team'] == 0 and np.array(obs['sticky_actions'])[8] == 1:  # ball owned and sprinting
                player_pos = obs['left_team'][obs['active']]
                goal_direction = (1 - player_pos[0]) * obs['ball_direction'][0]
                if goal_direction > 0:
                    components['counterattack_reward'][i] = self._counterattack_reward_factor

            # Aggregate all components and update the base reward
            total_additional_reward = components['tackling_reward'][i] + components['counterattack_reward'][i]
            reward[i] += total_additional_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
