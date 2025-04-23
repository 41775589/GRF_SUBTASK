import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specific reward function to aid our agents
    in learning defensive strategies such as tackling, shot-stopping,
    and retaining possession effectively.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
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
        """
        Customized reward function to enhance the defensive capabilities.
        Awards additional rewards for successful tackles, saves and retaining possession under pressure.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "tackle_reward": [0.0] * len(reward),
            "save_reward": [0.0] * len(reward),
            "possession_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Tackle rewards when the ball is owned by opponents near our goal
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player'] and abs(o['ball'][0]) > 0.5:
                components["tackle_reward"][i] += 0.5
            
            # Save rewards for the goalkeeper stopping a shot at the goal
            if o['right_team_roles'][o['active']] == 0 and o['game_mode'] == 6: # Assuming goalie index is 0
                components["save_reward"][i] += 1.0
            
            # Possession reward if managing to keep the ball under pressure
            if o['ball_owned_team'] == 0 and np.sum(o['sticky_actions'][8:10]) > 0:  # Sprint or dribble actions active
                components["possession_reward"][i] += 0.2
        
        for key in components:
            reward = [r + c for r, c in zip(reward, components[key])]

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
