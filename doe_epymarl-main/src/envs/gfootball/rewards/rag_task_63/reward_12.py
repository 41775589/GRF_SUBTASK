import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a tailored reward for training a goalkeeper."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define relevant parameters for custom reward modulations
        self.enemy_goals_area_trigger_distance = 0.1  # Proximity to trigger pressure situation handling

    def reset(self):
        """Reset the counter for sticky actions."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Get the state with added stored data for this wrapper."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state with specific unpacking for what this wrapper added."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        """Calculate custom reward for goalkeeper training."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "extra_rewards": [0.0] * len(reward)}
        
        # Default environment should provide list of observations
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Custom reward logic
            distance_to_goal = np.abs(o['ball'][0] - 1)  # X axis distance to the opponent's goal
            
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                if distance_to_goal < self.enemy_goals_area_trigger_distance:
                    # Reward the goalkeeper for being ready in a high-pressure situation
                    components["extra_rewards"][rew_index] = 0.5
                    active_player_role = o['right_team_roles'][o['active']]
                    
                    if active_player_role == 0:  # e_PlayerRole_GK
                        # Additional rewards if the goalkeeper is playing the ball correctly under pressure
                        if o['sticky_actions'][4] == 1:  # action_right, assuming simplification for positioning
                            reward[rew_index] += 2.0  # significant reward for ideal behavior
                        elif o['sticky_actions'][5] == 1 or o['sticky_actions'][7] == 1:
                            reward[rew_index] += 1.0  # lesser reward for other defensive actions
            # Finalize reward for this step
            reward[rew_index] += components["extra_rewards"][rew_index]
        
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
                self.sticky_actions_counter[i] += action
                
        return observation, reward, done, info
