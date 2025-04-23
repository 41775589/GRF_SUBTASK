import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes initiating counterattacks via accurate long passes and quick transitions
    from defense to attack. The reward is augmented when a long pass is successfully made from the defensive half
    to the attacking half, and when the transition speed (difference in positions of the player making the pass
    between two steps) is high.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_position': self.last_ball_position
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']['last_ball_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'transition_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Checking if a long pass is made by tracking ball position transitions.
            if 'ball' in o and 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Assuming team 0 is defense.
                current_ball_pos = o['ball'][0]
                
                # Transition from defense (-1 to 0 range) to attack (0 to 1 range)
                if self.last_ball_position is not None and self.last_ball_position <= 0 and current_ball_pos > 0:
                    components['transition_reward'][rew_index] = 1.0  # reward for moving the ball from defense to attack
                    
                self.last_ball_position = current_ball_pos
            
            # Calculate the reward based on the components
            reward[rew_index] += components['transition_reward'][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_flag in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_flag
        return observation, reward, done, info
