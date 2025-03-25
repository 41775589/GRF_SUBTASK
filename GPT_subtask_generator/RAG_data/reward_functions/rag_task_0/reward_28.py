import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a complex reward based on offensive gameplay skills: shooting, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Custom parameters to tweak the importance of different skills
        self.shooting_reward = 1.0
        self.dribbling_reward = 0.5
        self.passing_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'shooting_reward': self.shooting_reward, 'dribbling_reward': self.dribbling_reward, 'passing_reward': self.passing_reward}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.shooting_reward = state_info['shooting_reward']
        self.dribbling_reward = state_info['dribbling_reward']
        self.passing_reward = state_info['passing_reward']
        return from_pickle

    def reward(self, reward):
        observation = self.unwrapped.observation()
        base_score_reward = reward.copy()
        shooting_reward = [0.0] * len(reward)
        dribbling_reward = [0.0] * len(reward)
        passing_reward = [0.0] * len(reward)
        
        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            game_mode = obs['game_mode']
            ball_owned_team = obs['ball_owned_team']
            action_sprint = obs['sticky_actions'][8]  # Index 8 is sprint action
            action_dribble = obs['sticky_actions'][9]  # Index 9 is dribble action
            
            # Checking conditions for shooting at the goal
            if game_mode == 6 and ball_owned_team == 0:  # 6 is the game mode for shooting
                shooting_reward[idx] += self.shooting_reward
            
            # Handling dribbling reward
            if action_dribble:
                dribbling_reward[idx] += self.dribbling_reward
            
            # Handling passing reward in possession
            if ball_owned_team == 0 and (action_sprint or action_dribble):
                passing_reward[idx] += self.passing_reward

            # Sum up rewards
            reward[idx] = base_score_reward[idx] + shooting_reward[idx] + dribbling_reward[idx] + passing_reward[idx]
        
        reward_components = {
            'base_score_reward': base_score_reward,
            'shooting_reward': shooting_reward,
            'dribbling_reward': dribbling_reward,
            'passing_reward': passing_reward
        }
        
        return reward, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
