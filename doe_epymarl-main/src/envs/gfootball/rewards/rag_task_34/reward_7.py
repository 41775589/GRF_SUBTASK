import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on dribbling and shot precision close to the goal."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_distance_threshold = 0.1  # Distance from goal to consider 'close-range'
        self.dribble_reward_multiplier = 2.0  # Multiplier for dribbling close to the goal
        self.shot_reward_multiplier = 5.0  # Multiplier for precise shots close to the goal

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}
        
        new_reward = reward.copy()
        reward_components = {"base_score_reward": reward.copy(),
                             "dribble_reward": [0.0] * len(reward),
                             "shot_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            o = observation[i]

            # Calculate the distance to the opponent's goal (consider right team's perspective)
            distance_to_goal = abs(o['ball'][0] - 1)
            
            if o['game_mode'] == 0:  # Normal gameplay mode
                if distance_to_goal <= self.goal_distance_threshold:
                    if 'dribble' in o['sticky_actions'] and o['sticky_actions']['dribble']:
                        # Reward for dribbling close to the goal
                        reward_components['dribble_reward'][i] = self.dribble_reward_multiplier * (self.goal_distance_threshold - distance_to_goal)
                        new_reward[i] += reward_components['dribble_reward'][i]
            
                    if 'shot' in o['sticky_actions'] and o['sticky_actions']['shot']:
                        # Reward for shots close to the goal
                        reward_components['shot_reward'][i] = self.shot_reward_multiplier * (self.goal_distance_threshold - distance_to_goal)
                        new_reward[i] += reward_components['shot_reward'][i]

        return new_reward, reward_components

    def step(self, action):
        observations, reward, done, info = self.env.step(action)
        reward, reward_components = self.reward(reward)
        
        info['final_reward'] = sum(reward)
        for key, value in reward_components.items():
            info[f"component_{key}"] = sum(value)
            
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observations, reward, done, info

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle
