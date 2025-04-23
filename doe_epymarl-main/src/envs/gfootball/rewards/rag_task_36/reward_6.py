import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on improving dribbling maneuvers with dynamic positioning between defense and offense."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        if observation is None:
            return reward, components
        
        transition_reward = 0.0

        for index, obs in enumerate(observation):
            # Check if any dribbling action is active
            if obs['sticky_actions'][9] == 1:  # 'action_dribble' index 9
                # Encourage dribbling when changing from defensive to offensive positions and vice versa
                if (obs['left_team_direction'][obs['active']][0] * (obs['ball'][0] - observation[index]['left_team'][obs['active']][0]) > 0):
                    transition_reward += 0.05  # Player is dribbling towards correct half

            reward[index] += transition_reward
            components.setdefault('transition_reward', []).append(transition_reward)

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
            for i, action_status in enumerate(agent_obs.get('sticky_actions', [])):
                self.sticky_actions_counter[i] += action_status
        return observation, reward, done, info
