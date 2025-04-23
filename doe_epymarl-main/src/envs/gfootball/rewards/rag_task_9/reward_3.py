import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on offensive skills such as passing, shooting, and dribbling."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.05
        self.shot_reward = 0.1
        self.dribble_reward = 0.03
        self.sprint_reward = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return self.env.set_state(from_pickle)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for executing passes
            if o['sticky_actions'][0] == 1 or o['sticky_actions'][1] == 1:  # Short Pass, Long Pass
                components["pass_reward"][rew_index] = self.pass_reward
                reward[rew_index] += components["pass_reward"][rew_index]
            
            # Reward for shooting
            if o['game_mode'] == 6 and o['ball_owned_team'] == o['active']:  # Game mode 6 is Penalty
                components["shot_reward"][rew_index] = self.shot_reward
                reward[rew_index] += components["shot_reward"][rew_index]
            
            # Reward for dribbling with the ball
            if o['sticky_actions'][9] == 1 and o['ball_owned_team'] == o['active'] and o['ball_owned_player'] == o['active']:  # Dribble
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]
            
            # Reward for sprinting
            if o['sticky_actions'][8] == 1:  # Sprint
                components["sprint_reward"][rew_index] = self.sprint_reward
                reward[rew_index] += components["sprint_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Track sticky sprint/dribble actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
