import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive maneuvers and dynamic adaptation based on game phases."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.offensive_position_reward = 0.1
        self.dynamic_adaptation_reward = 0.05
        self.last_game_mode = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_game_mode = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointRewardWrapper'] = (self.sticky_actions_counter, self.last_game_mode)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter, self.last_game_mode = from_pickle['CheckpointRewardWrapper']
        return from_pickle
        
    def reward(self, reward):
        # Initialize reward components
        components = {'base_score_reward': reward.copy(), 'offensive_position_reward': [0] * len(reward),
                      'dynamic_adaptation_reward': [0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for getting the ball near the opponent's goal
            if o['ball_owned_team'] == 1:  # 1 indicates the right team in possession
                ball_position_x = o['ball'][0]
                if ball_position_x > 0.5:  # Ball is on the opponent's half
                    components['offensive_position_reward'][rew_index] = self.offensive_position_reward
            
            # Reward for adaptation based on game mode changes
            current_game_mode = o['game_mode']
            if self.last_game_mode is not None and self.last_game_mode != current_game_mode:
                components['dynamic_adaptation_reward'][rew_index] = self.dynamic_adaptation_reward

            # Aggregate rewards
            reward[rew_index] += (components['offensive_position_reward'][rew_index] +
                                   components['dynamic_adaptation_reward'][rew_index])
            
            # Update tracked game mode
            self.last_game_mode = current_game_mode

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Include details of reward components in the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Track sticky actions usage
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
