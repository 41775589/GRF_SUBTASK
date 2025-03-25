import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering long passes in football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.long_passes_completed = 0
        # distance thresholds to qualify as a long pass, in normalized field units
        self.long_pass_threshold = 0.5  
        self.long_pass_reward = 0.2  # reward for a successful long pass
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.long_passes_completed = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['long_passes_completed'] = self.long_passes_completed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.long_passes_completed = from_pickle['long_passes_completed']
        return from_pickle

    def reward(self, reward):
        """Calculate and allocate reward based on long passing criteria."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for idx in range(len(reward)):
            current_obs = observation[idx]
            
            # Check if the ball was passed and calculate distance
            if current_obs['ball_owned_team'] == 1:
                ball_change = np.linalg.norm(current_obs['ball_direction'][:2])
                if ball_change > self.long_pass_threshold:
                    components["long_pass_reward"][idx] = self.long_pass_reward
                    reward[idx] += self.long_pass_reward
                    self.long_passes_completed += 1

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
