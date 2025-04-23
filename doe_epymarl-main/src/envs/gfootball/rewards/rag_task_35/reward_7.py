import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrappper that focuses on maintaining strategic positioning, 
    using directional movements and balancing defense and attack."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy()}
        
        components = {'base_score_reward': reward.copy(), 'positional_reward': [0.0, 0.0]}
        
        for i in range(len(reward)):
            player_obs = observation[i]
            position, direction, ball_pos = player_obs['left_team'], player_obs['left_team_direction'], player_obs['ball']
            mid_line_x = 0.0

            # Check if player is behind the ball and moving towards it
            if position[i][0] < ball_pos[0] and direction[i][0] > 0:
                # Encourage moving towards the ball
                components['positional_reward'][i] += 0.05

            # Encourage staying back if the team does not have the ball
            if player_obs['ball_owned_team'] != 0 and position[i][0] < mid_line_x:
                components['positional_reward'][i] += 0.02

            # Reward for being between the ball and home goal if opponent team has the ball
            if player_obs['ball_owned_team'] == 1 and position[i][0] < ball_pos[0]:
                components['positional_reward'][i] += 0.02

            reward[i] += components['positional_reward'][i]
        
        return reward, components

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
