import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering midfield dynamics with emphasis on central and wide positions."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Track movement of midfielders across the field
        self.midfield_reach = {}
        self.midfield_reward_increase = 0.05
    
    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_reach = {}
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.midfield_reach
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_reach = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy()}

        for i, o in enumerate(observation):
            center_field_progress = self.calculate_midfield_progress(o)
            midfield_reward = (center_field_progress * self.midfield_reward_increase) - \
                               self.midfield_reach.get(i, 0)
            self.midfield_reach[i] = center_field_progress * self.midfield_reward_increase
            reward[i] += midfield_reward

        components["midfield_dynamic_reward"] = [self.midfield_reach.get(i, 0) for i in range(len(reward))]
        return reward, components
    
    def calculate_midfield_progress(self, obs):
        """Calculate progress based on midfield players' movement across the y-axis."""
        midfield_positions = []
        if 'left_team_roles' in obs:
            midfield_positions = obs['left_team'][obs['left_team_roles'] == 5]  # Central Midfield positions
            midfield_positions.extend(obs['left_team'][obs['left_team_roles'] == 6])  # Left Midfield positions
            midfield_positions.extend(obs['left_team'][obs['left_team_roles'] == 7])  # Right Midfield positions
            
        progress = np.mean([pos[0] for pos in midfield_positions])  # Focus on x-axis progress towards the opponent's goal
        return progress

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
