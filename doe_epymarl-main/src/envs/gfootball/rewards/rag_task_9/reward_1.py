import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for offensive football skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.actions_of_interest = {"short_pass": 0.1, "long_pass": 0.2, "shot": 0.3, "dribble": 0.1, "sprint": 0.05}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        return from_picle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), "skill_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                player_actions = o['sticky_actions']
                
                # Rewards for specific actions
                components["skill_reward"][rew_index] += sum(
                    player_actions[i] * val for i, val in enumerate([
                        self.actions_of_interest.get("sprint", 0),
                        self.actions_of_interest.get("dribble", 0),
                        self.actions_of_interest.get("short_pass", 0),
                        self.actions_of_interest.get("long_pass", 0),
                        self.actions_of_interest.get("shot", 0)
                    ])
                )
                
                reward[rew_index] += components["skill_reward"][rew_index]
        
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
