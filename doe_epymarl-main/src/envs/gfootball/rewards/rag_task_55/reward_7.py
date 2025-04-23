import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specific reward focused on tackling, intended to train agents in defensive tactics.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize the count of successful tackles, storing successful tackle positions.
        self.successful_tackles = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.successful_tackles = []
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.successful_tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.successful_tackles = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'tackle_reward': [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Get current game mode
            game_mode = o['game_mode']
            # Detect if it's normal play or a situation involving potential fouls
            if game_mode == 0:  # Normal mode
                ball_position = o['ball'][:2]
                ball_owned_team = o['ball_owned_team']
                active_player = o['active']
                is_tackling = o['sticky_actions'][7] or o['sticky_actions'][9]
                # Conditions where tackling ideally happens
                if ball_owned_team in [0, 1] and is_tackling:
                    player_position = o['left_team'][active_player] if ball_owned_team == 1 else o['right_team'][active_player]
                    distance_to_ball = np.linalg.norm(player_position - ball_position)
                    # Check if successful tackle without foul
                    if distance_to_ball < 0.03 and player_position not in self.successful_tackles:
                        components['tackle_reward'][rew_index] = 1.0
                        self.successful_tackles.append(player_position)
                        reward[rew_index] += components['tackle_reward'][rew_index]
        
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
