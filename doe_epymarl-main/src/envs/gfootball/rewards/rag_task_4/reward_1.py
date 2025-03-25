import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper to enhance dribbling and sprinting skills by rewarding
    tactical ball control, evasion from defenders, and effective usage of sprint actions.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
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
        components = {"base_score_reward": reward.copy(),
                      "dribbling_reward": [0.0] * len(reward),
                      "sprint_bonus": [0.0] * len(reward)}
        
        for i, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Ball with left team (assuming our agent is on the left)
                player_with_ball = o['active']
                
                # Encourage sprinting when moving forward with the ball in possession
                if o['sticky_actions'][8] == 1:  # Sprint action is index 8
                    components["sprint_bonus"][i] = 0.05  # Small constant reward for sprinting
                
                if player_with_ball == o['ball_owned_player']:
                    proximity_to_opponents = np.min(np.linalg.norm(o['left_team'][o['active']] - o['right_team'], axis=1))
                    
                    # Reward dribbling performance based on the proximity of opponents
                    if proximity_to_opponents < 0.1:  # Very close to opponents
                        components["dribbling_reward"][i] = 0.1
                    
                    # Raise reward if successfully maintaining ball control near opponents
                    if proximity_to_opponents < 0.05:
                        components["dribbling_reward"][i] += 0.2

            # Sum up rewards with components
            reward[i] += components["dribbling_reward"][i] + components["sprint_bonus"][i]
        
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
