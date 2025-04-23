import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper to encourage direct defensive actions and optimize response times to opponent attacks."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_successful = {}
        self.sliding_tackles_successful = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackles_successful = {}
        self.sliding_tackles_successful = {}
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "sliding_tackle_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['right_team'][o['active']]
            ball_pos = o['ball'][:2] # Getting the X, Y position
            
            # Check if the player is close to the ball to potentially make a tackle or sliding tackle
            distance_to_ball = np.linalg.norm(active_player_pos - ball_pos)
            
            if distance_to_ball < 0.05:  # Threshold to consider action impact on ball posseser
                if 'action_tackle' in o['sticky_actions'] and o['sticky_actions']['action_tackle']:
                    self.tackles_successful[rew_index] = self.tackles_successful.get(rew_index, 0) + 1
                    components["tackle_reward"][rew_index] = 0.5  # reward for each successful tackle

                if 'action_sliding' in o['sticky_actions'] and o['sticky_actions']['action_sliding']:
                    self.sliding_tackles_successful[rew_index] = self.sliding_tackles_successful.get(rew_index, 0) + 1
                    components["sliding_tackle_reward"][rew_index] = 1.0  # higher reward for more risky sliding tackle

            # Update the final reward considering base reward and additional rewards
            reward[rew_index] += (components["tackle_reward"][rew_index] + 
                                  components["sliding_tackle_reward"][rew_index]) 

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

    def get_state(self, to_pickle):
        to_pickle['tackles_successful'] = self.tackles_successful
        to_pickle['sliding_tackles_successful'] = self.sliding_tackles_successful
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.tackles_successful = from_pickle.get('tackles_successful', {})
        self.sliding_tackles_successful = from_pickle.get('sliding_tackles_successful', {})
        return from_pickle
