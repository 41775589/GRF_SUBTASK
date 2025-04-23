import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful tackles to regain ball possession without fouls."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_made = 0
        
        # Reward weights
        self.tackle_success_reward = 1.0
        self.foul_penalty = -2.0

    def reset(self):
        """Reset the environment and the tackle counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_made = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the current state with added tackles made info."""
        to_pickle['tackles_made'] = self.tackles_made
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state and retrieve tackles made info."""
        from_pickle = self.env.set_state(state)
        self.tackles_made = from_pickle['tackles_made']
        return from_pickle

    def reward(self, reward):
        """Enhance reward based on successful tackles."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "foul_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_possession_change = o['ball_owned_team'] == 1  # Assuming agent's team is 1

            if ball_possession_change:
                if 'tackle' in o['sticky_actions']:
                    components["tackle_reward"][rew_index] = self.tackle_success_reward
                    self.tackles_made += 1
                if 'foul' in o['sticky_actions']:
                    components["foul_penalty"][rew_index] = self.foul_penalty
                    
                reward[rew_index] += components["tackle_reward"][rew_index]
                reward[rew_index] += components["foul_penalty"][rew_index]

        return reward, components

    def step(self, action):
        """Execute step, compute extra rewards, return updated rewards and info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Track sticky actions for analysis
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action

        return observation, reward, done, info
