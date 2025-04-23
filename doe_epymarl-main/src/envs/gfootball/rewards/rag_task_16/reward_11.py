import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for technical skill enhancement in executing high passes with precision.
    Includes trajectory control, power assessment, and situational application drills.
    This will incentivize players to attempt and successfully complete high passes during game play.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters to adjust reward sensitivity
        self.pass_height_threshold = 0.15  # Height above which a pass is considered 'high'
        self.successful_pass_reward = 0.5  # Reward for successful high pass
        self.attempt_pass_reward = 0.1     # Small reward for just attempting a high pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """
        Adjust the original reward by adding specific rewards for high passes.
        """
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, {'base_score_reward': reward}

        components = {'base_score_reward': reward.copy(),
                      'high_pass_reward': [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Determine if a high pass has been attempted
            ball_z = o['ball'][2]  # Z-axis position of the ball
            possession = o['ball_owned_team'] == 1  # Assumes agent's team is '1'

            # Check if there is a high ball in play and own team is in possession
            if ball_z > self.pass_height_threshold and possession:
                components['high_pass_reward'][rew_index] += self.attempt_pass_reward
                reward[rew_index] += self.attempt_pass_reward
            
            # Additionally check if the high pass is effective and to a teammate
            if ball_z > self.pass_height_threshold and possession and 'designated' in o and o['designated'] == o['active']:
                components['high_pass_reward'][rew_index] += self.successful_pass_reward
                reward[rew_index] += self.successful_pass_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        processed_reward, components = self.reward(reward)
        info['final_reward'] = sum(processed_reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, processed_reward, done, info
