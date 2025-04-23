import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward based on dribbling and dynamic positioning."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_ownership = None
        self.dribble_scores = np.zeros(2)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_ownership = None
        self.dribble_scores = np.zeros(2)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_ball_ownership': self.previous_ball_ownership,
            'dribble_scores': self.dribble_scores,
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.previous_ball_ownership = from_pickle['CheckpointRewardWrapper']['previous_ball_ownership']
        self.dribble_scores = from_pickle['CheckpointRewardWrapper']['dribble_scores']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), "dynamic_positioning": [0.0]*2, "effective_dribbling": [0.0]*2}
        
        for agent_idx in range(2):
            agent_obs = observation[agent_idx]
            
            # Check dribbling status
            dribbling = agent_obs['sticky_actions'][9]  # 9 is the index for dribble action
            if dribbling and (self.previous_ball_ownership == agent_obs['active']):
                self.dribble_scores[agent_idx] += 0.05
            components["effective_dribbling"][agent_idx] = self.dribble_scores[agent_idx]

            # Dynamic position changes influencing transitions between defense and attack
            if agent_obs['ball_owned_team'] == 0: # 0 is the index for the left team, 1 for the right team
                dynamic_bonus = np.clip(np.linalg.norm(agent_obs['left_team'][agent_obs['active']] - agent_obs['ball']), 0, 1)
            else:
                dynamic_bonus = np.clip(np.linalg.norm(agent_obs['right_team'][agent_obs['active']] - agent_obs['ball']), 0, 1)
            
            components["dynamic_positioning"][agent_idx] = 0.1 / (dynamic_bonus + 0.1)  # Less distance gives higher score
        
            # Combine the effects
            reward[agent_idx] += components["dynamic_positioning"][agent_idx] + components["effective_dribbling"][agent_idx]
        
        self.previous_ball_ownership = observation[0]['ball_owned_player']
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
