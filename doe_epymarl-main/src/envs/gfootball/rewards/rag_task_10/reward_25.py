import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for extraordinary defensive actions."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # counts the defensive actions
        self.defensive_rewards = {
            'slide': 0.3,  # sliding tackles
            'intercept': 0.2,  # intercepting passes
            'stop_dribble': 0.1  # stopping dribble movements
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_rewards": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Defensive actions improved reward calculations
            if o['ball_owned_team'] == 1 and o['active'] in o['left_team']:
                # Extra reward for successfully stopping opponent dribble
                if 'stop_dribble' in self.defensive_rewards and o['sticky_actions'][9]:  # Using dribble action to infer stop-dribble
                    components['defensive_rewards'][rew_index] += self.defensive_rewards['stop_dribble']
                    reward[rew_index] += self.defensive_rewards['stop_dribble']
                
                # Extra reward for sliding tackles
                if 'slide' in self.defensive_rewards and o['sticky_actions'][8]:  # Using some action indexing for sliding
                    components['defensive_rewards'][rew_index] += self.defensive_rewards['slide']
                    reward[rew_index] += self.defensive_rewards['slide']
                
                # Reward for interception depends on the movement towards ball and ball's speed
                ball_speed = np.linalg.norm(o['ball_direction'][:2])
                player_speed = np.linalg.norm(o['left_team_direction'][o['active']][:2])
                if 'intercept' in self.defensive_rewards and ball_speed > 0.1 and player_speed > 0.1:
                    components['defensive_rewards'][rew_index] += self.defensive_rewards['intercept']
                    reward[rew_index] += self.defensive_rewards['intercept']

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
