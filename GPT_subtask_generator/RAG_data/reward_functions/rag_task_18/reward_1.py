import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward by evaluating midfield transition and pace control."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define additional parameters for rewards
        self.midfield_transition_bonus = 0.5
        self.pace_control_bonus = 0.3
        self.tired_penalty = -0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Fetch the latest environment observation
        observation = self.env.unwrapped.observation()
        
        # Initialize component structures
        components = {"base_score_reward": reward.copy(),
                      "midfield_transition_bonus": [0.0] * len(reward),
                      "pace_control_bonus": [0.0] * len(reward),
                      "tired_penalty": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for successful midfield transitions
            if 'left_team_roles' in o and 'right_team_roles' in o:
                # Reward central midfielders specifically, presumed indices 5 for CM
                if o['active'] in [5]:  # Assuming index 5 is central midfielder
                    # Check for control stability
                    if (np.abs(o['left_team_direction'][:, 0]).mean() < 0.01 and # Minimal horizontal movement
                        np.abs(o['left_team_direction'][:, 1]).mean() < 0.01):  # Minimal vertical movement
                        components["midfield_transition_bonus"][rew_index] = self.midfield_transition_bonus
                    
            # Reward for pace control
            if 'left_team_tired_factor' in o:
                # Penalize tiredness
                if o['left_team_tired_factor'].mean() > 0.5:  # More than half tired
                    components["tired_penalty"][rew_index] = self.tired_penalty
                else:
                    components["pace_control_bonus"][rew_index] = self.pace_control_bonus
            
            # Aggregating rewards
            reward[rew_index] += (components["midfield_transition_bonus"][rew_index] +
                                  components["pace_control_bonus"][rew_index] +
                                  components["tired_penalty"][rew_index])

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
