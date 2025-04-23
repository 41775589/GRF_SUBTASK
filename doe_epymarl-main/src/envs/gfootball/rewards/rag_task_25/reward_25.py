import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for dribbling techniques and sprints."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize configuration parameters.
        self.sprint_reward = 0.02
        self.dribble_reward = 0.03
        self.control_bonus = 0.05

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'sprint_reward': [0.0] * len(reward),
                      'dribble_reward': [0.0] * len(reward),
                      'control_bonus': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            action_sprint = obs['sticky_actions'][8]  # Index for 'sprint' action
            action_dribble = obs['sticky_actions'][9]  # Index for 'dribble' action

            # Assign rewards for sprinting and dribbling.
            if action_sprint:
                components['sprint_reward'][rew_index] = self.sprint_reward
            if action_dribble:
                components['dribble_reward'][rew_index] = self.dribble_reward

            # Control bonus when moving the ball forward under pressure.
            if obs['ball_owned_team'] == 0 or obs['ball_owned_team'] == 1:
                if np.linalg.norm(obs['ball_direction'][:2]) > 0.01:  # If the ball is moving
                    components['control_bonus'][rew_index] = self.control_bonus

            reward[rew_index] += sum(components[key][rew_index] for key in components)

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_ in enumerate(agent_obs['sticky_actions']):
                info[f'sticky_actions_{i}'] = action_
        return observation, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle
