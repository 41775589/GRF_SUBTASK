import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for defensive tasks, particularly focusing on 'stopping' roles such as intercepting passes, 
       blocking shots, and positioning effectively in man-marking scenarios."""
    
    def __init__(self, env):
        super().__init__(env)
        self.intercept_bonus = 0.5
        self.block_bonus = 0.5
        self.positioning_bonus = 0.2
        self.previous_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions = []

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        self.defensive_positions.clear()
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'previous_ball_owner': self.previous_ball_owner}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'CheckpointRewardWrapper' in from_pickle:
            self.previous_ball_owner = from_pickle['CheckpointRewardWrapper']['previous_ball_owner']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "intercept_reward": [0.0] * len(reward),
                      "block_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # High reward for intercepting the ball
            if o['ball_owned_team'] == o['left_team'] and self.previous_ball_owner and self.previous_ball_owner != o['ball_owned_team']:
                reward[rew_index] += self.intercept_bonus
                components["intercept_reward"][rew_index] = self.intercept_bonus
            
            # Additional reward for blocking shots
            if o['game_mode'] == 1:  # Assuming game_mode 1 stands for a shot at goal
                reward[rew_index] += self.block_bonus
                components["block_reward"][rew_index] = self.block_bonus

            # Reward for good positioning in man-marking based on proximity to a key opponent player
            opponent_positions = o['right_team'] if o['left_team'] else o['left_team']
            my_position = o['left_team'][rew_index] if o['left_team'] else o['right_team'][rew_index]
            distance = np.min(np.sqrt(np.sum((opponent_positions - my_position)**2, axis=1)))
            if distance < 0.1:  # arbitrary threshold for "close marking"
                reward[rew_index] += self.positioning_bonus
                components["positioning_reward"][rew_index] = self.positioning_bonus

            self.previous_ball_owner = o['ball_owned_team'] if 'ball_owned_team' in o else None

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
