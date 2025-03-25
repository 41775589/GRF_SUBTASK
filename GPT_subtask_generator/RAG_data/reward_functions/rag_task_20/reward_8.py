import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards offensive strategies, coordination, and reaction optimization."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_bonus = 0.05  # Reward for successful passes
        self.goal_distance_reward = 0.2  # Closer to goal positional advantage reward
    
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
        components = {
            "base_score_reward": reward.copy(),
            "pass_bonus": [0.0] * len(reward),
            "goal_distance_reward": [0.0] * len(reaction)
        }
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for idx, o in enumerate(observation):
            if 'sticky_actions' in o:
                pass_action = o['sticky_actions'][9]  # Assuming index 9 corresponds to pass action
                dribble_status = o['sticky_actions'][8]  # Assuming index 8 is dribble action
                if pass_action == 1:
                    reward[idx] += self.pass_bonus
                    components["pass_bonus"][idx] = self.pass_bonus

            # Calculate positional advantage based on distance from opponent's goal
            goal_pos = 1  # x-coordinate for opponent's goal position
            distance_to_goal = np.abs(goal_pos - o['right_team'][o['active']][0])
            proximity_reward = (1 - distance_to_goal) * self.goal_distance_reward
            reward[idx] += proximity_reward
            components["goal_distance_reward"][idx] = proximity_reward
            
            # General play reward modification for coordination and offense strategy
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                reward[idx] *= 1.1  # encouraging keeping the ball
            
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
