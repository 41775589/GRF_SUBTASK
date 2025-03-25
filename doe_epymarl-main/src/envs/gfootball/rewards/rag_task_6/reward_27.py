import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on stamina conservation strategies."""

    def __init__(self, env):
        super().__init__(env)
        # Constants for stamina and sprint incentives
        self.stamina_sprint_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Data structures for tracking rewards
        self.last_ball_ownership = None
        self.distance_traveled = [0.0, 0.0]

    def reset(self):
        # Reset sticky actions counters and distance metrics on reset
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.distance_traveled = [0.0, 0.0]
        self.last_ball_ownership = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'distance_traveled': self.distance_traveled.copy(),
            'last_ball_ownership': self.last_ball_ownership
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.distance_traveled = from_pickle['CheckpointRewardWrapper']['distance_traveled']
        self.last_ball_ownership = from_pickle['CheckpointRewardWrapper']['last_ball_ownership']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "stamina_sprint_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        for i, obs in enumerate(observation):
            if 'active' in obs and obs['active'] > -1:
                # Calculate increase in distance traveled when ball is owned
                if obs['ball_owned_team'] == 0:
                    if self.last_ball_ownership == obs['active']:
                        self.distance_traveled[obs['active']] += np.linalg.norm(obs['left_team_direction'][obs['active']])
                    else:
                        self.last_ball_ownership = obs['active']

                # Reward conservation of stamina using Sprint action sparingly
                using_sprint = obs['sticky_actions'][8]
                if not using_sprint and obs['right_team_tired_factor'][obs['active']] < 0.2:
                    # Reward if the player is not using sprint unnecessarily
                    components["stamina_sprint_reward"][i] = self.stamina_sprint_reward
                    reward[i] += components["stamina_sprint_reward"][i]
        
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
