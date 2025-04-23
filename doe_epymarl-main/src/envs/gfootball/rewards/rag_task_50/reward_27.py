import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for precise long passes, promoting better ball distribution.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_threshold = 0.3  # Distance threshold to consider a pass 'long'
        self.long_pass_reward = 0.2  # Reward weight for successful long passes

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        # Extract observations from environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Check if there was a significant change in ball position that is associated with a long pass
            if np.linalg.norm(o['ball_direction'][:2]) > self.pass_threshold and o['ball_owned_team'] in (0, 1):
                if self.previous_ball_pos is not None:  # Make sure there's a previous position to compare
                    distance = np.linalg.norm(self.previous_ball_pos - o['ball'][:2])
                    if distance >= self.pass_threshold:
                        # Ball moved a significant distance, reward the agent
                        components["long_pass_reward"][rew_index] = self.long_pass_reward
                        reward[rew_index] += self.long_pass_reward
            
            # Keep track of the last positions
            self.previous_ball_pos = o['ball'][:2]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        processed_reward, components = self.reward(reward)
        info["final_reward"] = sum(processed_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, processed_reward, done, info
