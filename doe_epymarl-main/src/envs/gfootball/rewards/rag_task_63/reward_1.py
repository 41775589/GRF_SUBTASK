import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a custom reward based on goalkeeper training tasks."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_interactions = 0
        self.ball_saves = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_interactions = 0
        self.ball_saves = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'ball_interactions': self.ball_interactions,
            'ball_saves': self.ball_saves
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_interactions = from_pickle['CheckpointRewardWrapper']['ball_interactions']
        self.ball_saves = from_pickle['CheckpointRewardWrapper']['ball_saves']
        return from_pickle

    def reward(self, reward):
        # Initialize component structure
        components = {
            "base_score_reward": reward.copy(),
            "save_reward": [0.0] * len(reward),
            "distribution_reward": [0.0] * len(reward)
        }
        
        # Retrieve the latest observation
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        # Calculate the rewards based on the goalkeeper's task
        for rew_index, o in enumerate(observation):
            if o['active'] == o['designated'] and o['ball_owned_team'] == 0:  # Check if controlled by agent and on ball owner
                # Reward successful saves
                if o['game_mode'] in [2, 3, 4]:  # Game modes related to potential scoring from opponents
                    self.ball_interactions += 1
                    components["save_reward"][rew_index] = 1.0
                    self.ball_saves += 1

                # Reward for distributing the ball effectively
                if np.linalg.norm(o['ball_direction']) > 0.5:  # Arbitrary threshold for effective kick
                    components["distribution_reward"][rew_index] = 0.5

                # Accumulate the rewards
                for key in components:
                    reward[rew_index] += components[key][rew_index]

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
