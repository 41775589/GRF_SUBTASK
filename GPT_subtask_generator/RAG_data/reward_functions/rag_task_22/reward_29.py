import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for sprint actions to improve defensive coverage.
    It rewards the agents for moving faster across the field using sprint actions,
    which is crucial for better defensive arrangements in dynamic game situations.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sprint_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self, **kwargs):
        self.sprint_actions_counter.fill(0)
        return self.env.reset(**kwargs)
        
    def get_state(self, to_pickle):
        to_pickle['sprint_actions_counter'] = self.sprint_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sprint_actions_counter = from_pickle['sprint_actions_counter']
        return from_pickle
    
    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "sprint_reward": [0.0] * len(reward)
        }
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            sprint_action = o['sticky_actions'][8]  # Index 8 is 'action_sprint'
            
            # Reward the agent for using sprint effectively.
            if sprint_action == 1:
                if self.sprint_actions_counter[rew_index] < 10:
                    components["sprint_reward"][rew_index] = 0.05  # Incremental reward for sprinting
                    self.sprint_actions_counter[rew_index] += 1
                else:
                    # Cap the total sprint rewards to avoid too much running without purpose
                    components["sprint_reward"][rew_index] = 0.0
            
            reward[rew_index] += components["sprint_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
