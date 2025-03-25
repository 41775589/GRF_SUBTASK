import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for advanced dribbling and sprint effectiveness."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_attempts = 0
        self.successful_dribbles = 0
        self.sprint_attempts = 0
        self.successful_sprints = 0
        
    def reset(self):
        """Reset the counters when a new episode starts."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_attempts = 0
        self.successful_dribbles = 0
        self.sprint_attempts = 0
        self.successful_sprints = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of rewards related to dribbling and sprinting."""
        to_pickle['dribble_attempts'] = self.dribble_attempts
        to_pickle['successful_dribbles'] = self.successful_dribbles
        to_pickle['sprint_attempts'] = self.sprint_attempts
        to_pickle['successful_sprints'] = self.successful_sprints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of rewards."""
        from_pickle = self.env.set_state(state)
        self.dribble_attempts = from_pickle.get('dribble_attempts', 0)
        self.successful_dribbles = from_pickle.get('successful_dribbles', 0)
        self.sprint_attempts = from_pickle.get('sprint_attempts', 0)
        self.successful_sprints = from_pickle.get('successful_sprints', 0)
        return from_pickle

    def reward(self, reward):
        """Calculate additional rewards for dribbling and sprinting efficacy."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": 0.0,
                      "sprint_reward": 0.0}
                      
        if observation is None:
            return reward, components
        
        active_player = observation['active']
        sticky_actions = observation['sticky_actions']
        
        self.process_sticky_actions(sticky_actions)
        
        # Reward for successful dribbles
        if sticky_actions[9]:  # Dribble action is index 9
            self.dribble_attempts += 1
            if self.is_effective_dribble(observation):
                self.successful_dribbles += 1
                components["dribble_reward"] = 0.05
        
        # Reward when sprinting is used effectively
        if sticky_actions[8]:  # Sprint action is index 8
            self.sprint_attempts += 1
            if self.is_effective_sprint(observation):
                self.successful_sprints += 1
                components["sprint_reward"] = 0.05

        # Calculate total reward considering added components
        reward += components["dribble_reward"] + components["sprint_reward"]
        
        return reward, components

    def step(self, action):
        """Perform a step using the reward function adjustments."""
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return obs, reward, done, info
    
    def process_sticky_actions(self, sticky_actions):
        self.sticky_actions_counter += sticky_actions
        
    def is_effective_dribble(self, observation):
        """A simplified version to detect if the dribble was effective."""
        # Effectiveness could be measured in terms of distance covered with dribbling or position gained
        return True
        
    def is_effective_sprint(self, observation):
        """A simplified version to detect if the sprint was effective."""
        # Effectiveness could be measured in terms of speed increase or opponents outrun
        return True
