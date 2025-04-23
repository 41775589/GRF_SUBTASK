import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """ A wrapper that provides a dense reward function tailored for goalkeeper training. """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "save_bonus": [0.0] * len(reward),
                      "kick_distance_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        # Assuming goalkeeper is part of right_team for example purpose and there are two agents (left and right teams).
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Goalkeeping and reflex rewards
            if 'right_team_roles' in o and o['active'] == 0 and o['right_team_roles'][o['active']] == 0:  # assuming index 0 is the goalkeeper
                if o['ball_owned_team'] == 1 and o['ball_owned_player'] in [o['active']]:
                    components["save_bonus"][rew_index] = 1.0
                    reward[rew_index] += components["save_bonus"][rew_index]
            
            # Rewards for initiating attacks (passing accuracy and distance)
            if 'ball_direction' in o and o['ball_owned_team'] == 1:
                distance_to_goal_x = 1 - o['ball'][0]  # the x position of the goal is 1
                if 'ball_owned_player' in o and o['ball_owned_player'] in [o['active']]:
                    components["kick_distance_bonus"][rew_index] = distance_to_goal_x * 0.5  # 0.5 is just a coefficient
                    reward[rew_index] += components["kick_distance_bonus"][rew_index]

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
