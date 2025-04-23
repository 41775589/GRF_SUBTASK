import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A gym wrapper that adds rewards based on ball control and successful passing 
    under tight game situations, focusing on Short Pass, High Pass, and Long Pass.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_bonus = 0.2
        self.control_under_pressure_bonus = 0.3
        self.reset()

    def reset(self):
        """
        Reset the environment and reset sticky actions counter.
        """
        self.sticky_actions_counter.fill(0)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """
        Include wrapper's state into the pickle.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """
        Restore wrapper's state from the pickle.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Augment the original reward based on advanced ball control and passing under pressure.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_completion_bonus": [0.0] * len(reward),
                      "control_under_pressure_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Check for control under pressure if ball is owned by the active player
            if o['ball_owned_team'] == o['sticky_actions'][8] and o['ball_owned_player'] == o['active']:
                opponents_near = np.sum(np.linalg.norm(o['right_team'][:2] - o['ball'][:2], axis=1) < 0.1)
                if opponents_near > 0:
                    components["control_under_pressure_bonus"][rew_index] += self.control_under_pressure_bonus
                    reward[rew_index] += components["control_under_pressure_bonus"][rew_index]
            
            # Evaluate successful pass based on sticky actions corresponding to Short Pass, High Pass, and Long Pass
            # Assuming successful pass is triggered by a positive ball direction change during corresponding actions
            if (o['sticky_actions'][0] == 1 or o['sticky_actions'][1] == 1) and np.linalg.norm(o['ball_direction']) > 0:
                components["pass_completion_bonus"][rew_index] += self.pass_completion_bonus
                reward[rew_index] += components["pass_completion_bonus"][rew_index]
        
        return reward, components

    def step(self, action):
        """
        Take an environment step and compute custom reward modifications.
        """
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
