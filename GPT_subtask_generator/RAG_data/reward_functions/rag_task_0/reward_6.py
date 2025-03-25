import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive strategies in football.
    
    This includes rewards for accurate shooting, effective dribbling, and practicing varied pass types.
    It emphasizes improving offensive gameplay such as maintaining possession, progressing towards opponent's goal,
    and executing shots and passes effectively.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_checkpoint = 0.3  # Reward for successful long or high passes
        self.shot_accuracy = 0.5    # Reward for accurate shots on goal
        self.dribble_effectiveness = 0.2  # Reward for successful dribbling past an opponent
        self.ball_progression = 0.1 # Reward for moving the ball towards the opponent's goal
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
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_checkpoint": [0.0] * len(reward),
            "shot_accuracy": [0.0] * len(reward),
            "dribble_effectiveness": [0.0] * len(reward),
            "ball_progression": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for idx, obs in enumerate(observation):
            # Check if the agent made a successful shot
            if obs['game_mode'] == 6 and obs['ball_owned_team'] == 0:  # Assuming 6 is the shot mode
                components['shot_accuracy'][idx] = self.shot_accuracy
                reward[idx] += components['shot_accuracy'][idx]
            
            # Check if the agent has successfully dribbled by checking sticky actions
            if obs['sticky_actions'][9] == 1:  # Assuming action 9 is dribble
                components['dribble_effectiveness'][idx] = self.dribble_effectiveness
                reward[idx] += components['dribble_effectiveness'][idx]
            
            # Reward for passing (considering long and high passes)
            if 'long_pass' in obs and obs['long_pass'] or 'high_pass' in obs and obs['high_pass']:
                components['pass_checkpoint'][idx] = self.pass_checkpoint
                reward[idx] += components['pass_checkpoint'][idx]
            
            # Ball progression towards opponent's goal by checking the x-coordinate of the ball position
            if obs['ball'][0] > 0:  # Assuming positive x-axis is towards opponent's goal
                components['ball_progression'][idx] = self.ball_progression * obs['ball'][0]
                reward[idx] += components['ball_progression'][idx]
        
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
