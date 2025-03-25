import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that enhances the synergistic effectiveness of 
    central midfield by focusing on seamless transitions and controlled 
    pace management."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters that can be tuned for different intermediate rewards
        self.transition_reward = 0.5
        self.pace_control_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        from_pickle = self.env.get_state(to_pickle)
        return from_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "transition_reward": [0.0] * len(reward),
            "pace_control_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            # Focusing transition reward dynamics
            ball_owned_team = obs['ball_owned_team']
            ball_owned_player = obs['ball_owned_player']
            if ball_owned_team == 0 and ball_owned_player == obs['active']:
                components['transition_reward'][i] += self.transition_reward

            # Incorporating pace control
            if obs['left_team_tired_factor'][obs['active']] < 0.1:
                components['pace_control_reward'][i] += self.pace_control_reward

            # Total reward combining all components for agent i
            reward[i] = (reward[i] +
                         components['transition_reward'][i] +
                         components['pace_control_reward'][i])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        # Recording reward components for analysis
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
