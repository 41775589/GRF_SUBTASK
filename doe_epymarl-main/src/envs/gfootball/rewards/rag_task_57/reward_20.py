import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances reward signals for mastering offensive strategies between midfielders and strikers."""

    def __init__(self, env):
        super().__init__(env)
        self.midfielder_positions = []
        self.striker_positions = []
        self.ball_delivery_reward = 0.5
        self.finish_play_reward = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['midfielder_positions'] = self.midfielder_positions
        state['striker_positions'] = self.striker_positions
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfielder_positions = from_pickle.get('midfielder_positions', [])
        self.striker_positions = from_pickle.get('striker_positions', [])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "ball_delivery_reward": [0.0] * len(reward),
                      "finish_play_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for i, o in enumerate(observation):
            # Check if the active player is in the midfield and has the ball.
            if o['active'] in self.midfielder_positions and o['ball_owned_team'] == 1:
                components['ball_delivery_reward'][i] = self.ball_delivery_reward
                reward[i] += components['ball_delivery_reward'][i]

            # Check if the active player is a striker and finishes a play.
            if o['active'] in self.striker_positions and o['score'][0] == 1:  # Assuming this team is left
                components['finish_play_reward'][i] = self.finish_play_reward
                reward[i] += components['finish_play_reward'][i]
            
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        observation = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
