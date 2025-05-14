import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper designed to specifically focus on improving sliding techniques and reactions to physical confrontations in a football game scenario."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sliding_counter = 0  # Adds a sliding counter to track sliding actions
        self.physical_confrontations_tracker = {}  # A dictionary to track confrontations context
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sliding_counter = 0
        self.physical_confrontations_tracker = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sliding_counter': self.sliding_counter,
            'physical_confrontations_tracker': self.physical_confrontations_tracker
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.sliding_counter = state_info['sliding_counter']
        self.physical_confrontations_tracker = state_info['physical_confrontations_tracker']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sliding_reward": [0.0],
                      "confrontation_reward": [0.0]}

        if observation is None:
            return reward, components
        
        o = observation[0]  # The observations of our single agent.

        # Identifies and rewards sliding actions, especially during physical confrontations.
        if o['sticky_actions'][6] == 1:  # Check if sliding (index 6) is active.
            if o['game_mode'] in [2, 3, 4, 6]:  # Modes potentially involving confrontations.
                # Increasing reward when sliding during confrontations.
                components["sliding_reward"][0] = 1.0
                components["confrontation_reward"][0] = 0.5
                self.physical_confrontations_tracker[o['steps_left']] = True
            self.sliding_counter += 1
        
        # Modify the reward by adding any external sliding and confrontation context understandings.
        reward[0] += sum(components["sliding_reward"]) + sum(components["confrontation_reward"])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        # Tracking sticky actions to provide additional context in the info dictionary.
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
