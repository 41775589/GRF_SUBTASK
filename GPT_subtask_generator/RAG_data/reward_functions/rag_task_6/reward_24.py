import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards energy conservation strategies by monitoring use of Stop-Sprint and Stop-Moving actions."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # To count activations of each sticky action

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Reset action counts on new episode
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "energy_conservation_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'sticky_actions' in o:
                # Get use of stop sprint (action index 8) and stop moving (no movement action index 0)
                stop_sprint_active = o['sticky_actions'][8] 
                stop_move_active = o['sticky_actions'][0]
                
                # Encourage reducing the use of sprints and movement when not necessary
                if stop_sprint_active or stop_move_active:
                    components["energy_conservation_bonus"][rew_index] = 0.05  # Small constant reward for stopping sprint and move
                    reward[rew_index] += components["energy_conservation_bonus"][rew_index]
                    
            # Update the sticky actions count
            self.sticky_actions_counter += o['sticky_actions']
        
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
