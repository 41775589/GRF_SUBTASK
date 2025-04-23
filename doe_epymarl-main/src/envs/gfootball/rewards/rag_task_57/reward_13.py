import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward based on coordinated offensive plays."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.num_midfielders = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_position = (0.5, 0)  # Approx mid-field position
        self.offensive_threshold = 0.7  # Position x > 0.7 is considered offensive
        self.midfielder_striker_coordination_bonus = 0.2

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
        components = {"base_score_reward": reward.copy(),
                      "coordination_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        midfielder_positions = []
        
        for rew_index, o in enumerate(observation):
            # Identify midfielders based on roles and gather their field positions.
            if o['left_team_roles'][o['active']] in {4, 5, 6}:  # Assuming DM, CM, LM roles ids
                if np.abs(o['left_team'][o['active']][0] - self.midfield_position[0]) < 0.1:
                    midfielder_positions.append(o['left_team'][o['active']])

        for rew_index, o in enumerate(observation):
            # Look for strikers (assuming role id 9 for CF)
            if o['left_team_roles'][o['active']] == 9:
                striker_position = o['left_team'][o['active']]
                for m_pos in midfielder_positions:
                    # Check if midfielder is well positioned with respect to striker
                    if striker_position[0] > self.offensive_threshold and np.linalg.norm(np.array(striker_position)-np.array(m_pos)) < 0.4:
                        components["coordination_reward"][rew_index] += self.midfielder_striker_coordination_bonus
                        reward[rew_index] += components["coordination_reward"][rew_index]                        

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
