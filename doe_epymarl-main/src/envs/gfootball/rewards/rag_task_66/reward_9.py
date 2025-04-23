import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on short passes under defensive pressure."""
    
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
        
        if observation is None:
            return reward, {}
        
        assert len(reward) == len(observation)

        components = {"base_score_reward": reward.copy(),
                      "pass_under_pressure": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            components["base_score_reward"][rew_index] = reward[rew_index]

            # Evaluate reward for successful short passes under pressure
            if o['game_mode'] == 0:  # Normal gameplay
                if o['ball_owned_team'] == 0 and 'action' in o:  # Assuming 0 is team ID of controlling agent
                    if o['action'] == 'short_pass':
                        # Relative position of players to the ball and evaluating defensive pressure
                        ball_pos = o['ball'][:2]
                        teammates_pos = [pos for i, pos in enumerate(o['left_team']) if i != o['active']]
                        opponents_pos = o['right_team']
                        
                        # Calculate distance to nearest opponent and teammate
                        min_opponent_dist = np.min([np.linalg.norm(ball_pos - pos) for pos in opponents_pos])
                        min_teammate_dist = np.min([np.linalg.norm(ball_pos - pos) for pos in teammates_pos])

                        # Additional reward if pass is made under pressure and distance to teammate is reasonable
                        if min_opponent_dist < 0.05 and min_teammate_dist < 0.2:
                            components["pass_under_pressure"][rew_index] = 0.1  # extra reward for performance under pressure
                        
            # Sum up various components of reward
            reward[rew_index] = (reward[rew_index] +
                                 components["pass_under_pressure"][rew_index])
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        # Adding additional information to 'info'
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
