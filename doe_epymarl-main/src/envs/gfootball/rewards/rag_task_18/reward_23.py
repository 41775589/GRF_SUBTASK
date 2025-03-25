import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward based on midfield effectiveness in managing transitions and pace control."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Setup the levels of midfield control needed, each level requiring increasing coordination between players
        self.midfield_control_levels = 5
        self.midfield_control_rewards = np.linspace(0.1, 0.5, self.midfield_control_levels)
        self.midfield_control_achieved = [False] * self.midfield_control_levels

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_control_achieved = [False] * self.midfield_control_levels
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.midfield_control_achieved
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_control_achieved = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        ball_position = observation['ball'][:2]  # Get XY only
        midfielders = [i for i, role in enumerate(observation['left_team_roles']) if role == 5]  # CM role is 5
        
        # Calculate midfield control based on position and possession
        controlled_midfielders = [i for i in midfielders if observation['left_team_active'][i] and observation['ball_owned_team'] == 0 and observation['ball_owned_player'] == i]
        active_midfield_control = len(controlled_midfielders) / len(midfielders)
        
        # Assign incremental rewards for better midfield control
        for index, threshold in enumerate(np.linspace(0, 1, self.midfield_control_levels, endpoint=False)):
            if active_midfield_control > threshold and not self.midfield_control_achieved[index]:
                components['midfield_bonus'][0] += self.midfield_control_rewards[index]
                self.midfield_control_achieved[index] = True
        
        reward[0] += sum(components['midfield_bonus'])
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
        return observation, reward, done, info
