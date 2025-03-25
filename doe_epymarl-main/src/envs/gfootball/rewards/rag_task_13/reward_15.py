import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds specific rewards for man-marking defense actions in a football game. 
    Rewards are based on blocking shots, intercepting passes, and positioning to stall forward moves.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = {}
        self.interception_reward = 0.2
        self.blocking_reward = 0.3
        self.positioning_reward = 0.1
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_positions'] = self.defensive_positions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self.defensive_positions = from_picle['defensive_positions']
        return from_picle

    def reward(self, reward):
        new_rewards = reward.copy()
        components = {'base_score_reward': reward.copy(), 'interception_reward': [0] * len(reward), 'blocking_reward': [0] * len(reward), 'positioning_reward': [0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return new_rewards, components
        
        assert len(new_rewards) == len(observation)

        for i, obs in enumerate(observation):
            components['base_score_reward'][i] = reward[i]
            
            if obs.get('ball_owned_team') == 1:  # Assuming team 1 is the opponent
                ball_position = obs['ball']
                if obs['game_mode'] == 0:  # In-play
                    defenders = np.where(obs['left_team_roles'] == 1)[0]  # Assuming 1 denotes defenders
                    distances = np.linalg.norm(obs['left_team'][defenders] - ball_position[:2], axis=1)
                    if np.any(distances < 0.1):  # Close enough to intercept or block shots
                        new_rewards[i] += self.interception_reward
                        components['interception_reward'][i] = self.interception_reward

            player_pos = obs['left_team'][obs['active']]
            desired_position = player_pos + obs['left_team_direction'][obs['active']]
            ball_dir = obs['ball_direction'][:2]
            if np.dot(desired_position, ball_dir) > 0:  # Is player positioning to stall forward moves?
                new_rewards[i] += self.positioning_reward
                components['positioning_reward'][i] = self.positioning_reward

            # Blocking simulated by checking if active player's intercept vector correlates highly with ball direction
            if obs['game_mode'] == 4:  # Freekicks or corners
                if np.linalg.norm(ball_dir - obs['ball_owned_player']) < 0.2:
                    new_rewards[i] += self.blocking_reward
                    components['blocking_reward'][i] = self.blocking_reward

        return new_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_reward, components = self.reward(reward)
        info['final_reward'] = sum(new_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for index, act in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{index}"] = act
        return observation, new_reward, done, info
