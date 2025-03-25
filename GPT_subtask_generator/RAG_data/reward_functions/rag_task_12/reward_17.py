import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides a reward for successfully handling mid-field control and switching,
    emphasizing high passes, long passes, dribbling under pressure, and effective sprint management."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.high_pass_counter = 0
        self.long_pass_counter = 0
        self.dribble_counter = 0
        self.sprint_counter = 0
        self.stop_sprint_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_counter = 0
        self.long_pass_counter = 0
        self.dribble_counter = 0
        self.sprint_counter = 0
        self.stop_sprint_counter = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "high_pass_counter": self.high_pass_counter,
            "long_pass_counter": self.long_pass_counter,
            "dribble_counter": self.dribble_counter,
            "sprint_counter": self.sprint_counter,
            "stop_sprint_counter": self.stop_sprint_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_info = from_pickle['CheckpointRewardWrapper']
        self.high_pass_counter = state_info["high_pass_counter"]
        self.long_pass_counter = state_info["long_pass_counter"]
        self.dribble_counter = state_info["dribble_counter"]
        self.sprint_counter = state_info["sprint_counter"]
        self.stop_sprint_counter = state_info["stop_sprint_counter"]
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": 0.0,
                      "long_pass_reward": 0.0,
                      "dribble_reward": 0.0,
                      "sprint_reward": 0.0,
                      "stop_sprint_reward": 0.0}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            o = observation[idx]
            components_idx = components.copy()
            
            if o['sticky_actions'][6]:  # High pass
                if self.high_pass_counter < 5:
                    self.high_pass_counter += 1
                    components_idx["high_pass_reward"] = 0.1
                    
            if o['sticky_actions'][8]:  # Long pass
                if self.long_pass_counter < 5:
                    self.long_pass_counter += 1
                    components_idx["long_pass_reward"] = 0.1

            if o['sticky_actions'][9]:  # Dribble
                if self.dribble_counter < 10:
                    self.dribble_counter += 1
                    components_idx["dribble_reward"] = 0.05
                    
            if o['sticky_actions'][5]:  # Sprint
                if self.sprint_counter < 10:
                    self.sprint_counter += 1
                    components_idx["sprint_reward"] = 0.02
            elif self.sticky_actions_counter[5] == 1:  # Stop Sprint
                if self.stop_sprint_counter < 10:
                    self.stop_sprint_counter += 1
                    components_idx["stop_sprint_reward"] = 0.02

            extras = sum(components_idx.values())
            reward[idx] += extras
            components[idx] = components_idx
            
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
