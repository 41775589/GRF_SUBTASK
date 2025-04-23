import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for mastering Stop-Sprint and Stop-Moving techniques.
    The agents are rewarded for effectively stopping from sprints and moving states,
    and starting to sprint or move again based on tactical game scenarios.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        # Basic reward input from the environment
        components = {
            "base_score_reward": reward.copy()
        }

        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components
        
        # Assume index 0 corresponds to the agent we are interested in.
        o = observation[0]

        # Checking if the agent has stopped sprinting this step
        if o['sticky_actions'][8] == 0 and self.sticky_actions_counter[8] > 0:
            reward[0] += 0.1  # Reward for stopping sprint
            self.sticky_actions_counter[8] = 0
        
        # Checking if the agent starts sprinting
        if o['sticky_actions'][8] == 1 and self.sticky_actions_counter[8] == 0:
            reward[0] += 0.1  # Reward for starting sprint
            self.sticky_actions_counter[8] = 1

        # Checking if the agent has stopped moving
        moving_actions = [0, 1, 2, 3, 4, 5, 6, 7]  # Left movements' indices
        if all(o['sticky_actions'][action] == 0 for action in moving_actions) and self.sticky_actions_counter[:8].sum() > 0:
            reward[0] += 0.1  # Reward for stopping moving
            self.sticky_actions_counter[:8] = 0
        
        # Checking if the agent starts moving
        if any(o['sticky_actions'][action] == 1 for action in moving_actions) and self.sticky_actions_counter[:8].sum() == 0:
            reward[0] += 0.1  # Reward for starting moving
            self.sticky_actions_counter[:8] = 1
        
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
            for i, action_flag in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action_flag
        return observation, reward, done, info
