import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on defensive strategies and midfield management."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Custom properties to track midfield and defensive maneuvers
        self.midfield_control_checkpoints = 5
        self.defensive_actions_counter = np.zeros(5, dtype=int)  # 5 zones to track for defensive actions
        self.midfield_actions_counter = np.zeros(5, dtype=int)  # 5 zones for midfield control

        self.defensive_threshold = 0.3  # sensitivity of defensive actions
        self.midfield_threshold = 0.3   # sensitivity of midfield actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_actions_counter.fill(0)
        self.midfield_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'defensive_actions': self.defensive_actions_counter,
                                                'midfield_actions': self.midfield_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_actions_counter = from_pickle['CheckpointRewardWrapper']['defensive_actions']
        self.midfield_actions_counter = from_pickle['CheckpointRewardWrapper']['midfield_actions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "midfield_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'right_team' in o:
                # Assuming right_team are the opponents and left_team are controlled
                right_team_dist_to_goal = np.min(np.sqrt(np.sum(np.square(o['right_team'][:, :2] - [1, 0]), axis=1)))
                left_team_dist_to_midfield = np.min(np.sqrt(np.sum(np.square(o['left_team'][:, :2] - [0, 0]), axis=1)))
                
                # Check for defensive actions close to own goal (x<=-0.5)
                if right_team_dist_to_goal < self.defensive_threshold and o['ball_owned_team'] == 0:
                    components["defensive_reward"][rew_index] = 0.05
                    self.defensive_actions_counter[rew_index] += 1
                
                # Check for midfield control
                if left_team_dist_to_midfield < self.midfield_threshold and o['ball_owned_team'] == 0:
                    components["midfield_reward"][rew_index] = 0.05
                    self.midfield_actions_counter[rew_index] += 1

                # Aggregate individual rewards
                reward[rew_index] += components["defensive_reward"][rew_index] + components["midfield_reward"][rew_index]

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
