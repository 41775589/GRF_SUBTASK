import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the training of defensive strategies, focusing on goalkeeper shot-stopping,
    and defenders' tackling and possession skills."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize sticky actions counter
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize rewards for specific defensive accomplishments
        self.goalkeeper_save_reward = 0.5
        self.defender_tackle_reward = 0.3
        self.possession_maintained_reward = 0.2
        
    def reset(self):
        """Resets the sticky actions counter on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Stores the state of the environment."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Recalls the state of the environment previously stored."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Adjusts the reward based on specific defensive actions."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_save_reward": [0.0] * len(reward),
                      "defender_tackle_reward": [0.0] * len(reward),
                      "possession_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Check if ball is with left team
                if 'ball_owned_player' in o:
                    if 'left_team_roles' in o and o['left_team_roles'][o['ball_owned_player']] == 0:  # Goalkeeper role
                        components["goalkeeper_save_reward"][rew_index] = self.goalkeeper_save_reward
                    elif o['left_team_roles'][o['ball_owned_player']] in [1, 2, 3, 4]:  # Defender roles
                        distance_to_ball = np.linalg.norm(o['ball'][:-1] - o['left_team'][o['ball_owned_player']])
                        if distance_to_ball < 0.1:  # Tackling distance threshold
                            components["defender_tackle_reward"][rew_index] = self.defender_tackle_reward
                        # Reward for maintaining possession under pressure
                        if o['sticky_actions'][9]:  # Check for dribble action active
                            components["possession_reward"][rew_index] = self.possession_maintained_reward
            
            total_rewards = (np.array(components["base_score_reward"]) +
                             np.array(components["goalkeeper_save_reward"]) +
                             np.array(components["defender_tackle_reward"]) +
                             np.array(components["possession_reward"]))
            
            reward[rew_index] = total_rewards[rew_index]
        
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
