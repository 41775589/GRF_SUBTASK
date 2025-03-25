import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for executing high passes with precision."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Probable score impacts when a high pass is executed precisely
        self.pass_success_threshold = 0.85  # assuming normalized distance measure
        self.high_pass_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Load any necessary state here if required
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "precision_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Assume that a strategy for executing high pass is implemented as a sticky action
            current_action = o['sticky_actions'][9]  # example index for 'pass' action

            # Calculate the efficiency of the pass
            if current_action == 1:  # assuming '1' denotes the ongoing high pass action
                ball_position = o['ball']  # ball position as [x, y, z]
                goal_position = [1, 0, 0]  # hypothetical position of the goal

                # Calculate Euclidean distance to the goal
                distance = np.linalg.norm(np.array(ball_position) - np.array(goal_position))

                # Check if the pass is precise and efficient
                if distance < self.pass_success_threshold:
                    components["precision_pass_reward"][rew_index] = self.high_pass_reward
                    reward[rew_index] += components["precision_pass_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Update sticky actions info
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
