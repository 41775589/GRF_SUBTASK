import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes goalkeeper training focused on shot-stopping, quick reflexes, 
    and initiating counter-attacks with accurate passes.
    """
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.quick_reflex_bonus = 1.0
        self.accurate_pass_bonus = 0.5
        self.shot_stopping_bonus = 2.0
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "quick_reflex_bonus": [0.0] * len(reward),
                      "accurate_pass_bonus": [0.0] * len(reward),
                      "shot_stopping_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index, o in enumerate(observation):
            # Assuming goalkeeper is always the first player in left_team when active (rew_index == 0)
            if o['active'] == 0 and o['left_team_roles'][0] == 0:  # 0 corresponds to Goalkeeper role
                # Detect shot stopping
                if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball'] - o['left_team'][0]) < 0.1:
                    components["shot_stopping_bonus"][rew_index] = self.shot_stopping_bonus
                    reward[rew_index] += self.shot_stopping_bonus

                # Check for quick reflex action: abrupt change in ball direction close to GK
                if np.linalg.norm(o['ball_direction']) > 0.5 and np.linalg.norm(o['ball'] - o['left_team'][0]) < 0.2:
                    components["quick_reflex_bonus"][rew_index] = self.quick_reflex_bonus
                    reward[rew_index] += self.quick_reflex_bonus

                # Bonus for initiating a counter-attack with accurate pass
                # Check if ball is owned by GK and a pass action (movement away from GK position)
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == 0 and np.dot(o['ball_direction'], o['left_team_direction'][0]) > 0:
                    components["accurate_pass_bonus"][rew_index] = self.accurate_pass_bonus
                    reward[rew_index] += self.accurate_pass_bonus

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
