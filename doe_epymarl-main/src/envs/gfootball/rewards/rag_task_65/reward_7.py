import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on shooting and passing effectiveness and strategic positioning."""
    
    def __init__(self, env):
        super().__init__(env)
        self.passing_accuracy_reward = 0.1
        self.shooting_accuracy_reward = 0.2
        self.positioning_reward = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        """Reset the environment and sticky_actions_counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        """Get the state of the environment with additional wrapper state."""
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        """Set the state of the environment and update sticky_actions_counter."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle
    
    def reward(self, reward):
        """Modify the reward based on passing and shooting accuracy and strategic positioning."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "shooting_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for accurate passes
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0: # team 0 implies our agent's team
                if np.any(o['sticky_actions'][6:8]): # 6: pass, 7: high pass
                    components['passing_reward'][rew_index] += self.passing_accuracy_reward

            # Reward for shots on target
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                if o['sticky_actions'][9]: # 9: shot
                    components['shooting_reward'][rew_index] += self.shooting_accuracy_reward
            
            # Strategic Positioning: reward staying close to goal-scoring areas
            if o['right_team_roles'][o['active']] == 9: # CF role
                close_to_goal_x = abs(o['right_team'][o['active']][0]) > 0.8
                components['positioning_reward'][rew_index] += self.positioning_reward if close_to_goal_x else 0
            
            # Aggregate the total reward
            reward[rew_index] = reward[rew_index] + components['passing_reward'][rew_index] + components['shooting_reward'][rew_index] + components['positioning_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Step through environment, modify reward, and record info about all components of the reward."""
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
