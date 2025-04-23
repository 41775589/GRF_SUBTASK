import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper to enhance mid to long-range passing effectiveness.
    This aims to reward effective pass execution and strategic orientation.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Rewards for passing precision and strategic use of passes
        self.pass_reward = 0.2
        self.strategic_pass_reward = 0.3

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
                      "pass_reward": [0.0] * len(reward),
                      "strategic_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' in o:
                # Evaluate long-range and strategic passes:
                # If ball owned by left team and active agent is in possession
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                    pass_quality = self.evaluate_pass_quality(o)
                    if pass_quality > 0.75:  # Consider a pass effective if it is over 75% precision
                        components["pass_reward"][rew_index] = self.pass_reward
                        reward[rew_index] += components["pass_reward"][rew_index]
                    
                    # Strategic passes could include forward and long distance towards goal
                    if o['ball_direction'][0] > 0.1 and np.linalg.norm(o['ball_direction']) > 0.5:
                        components["strategic_pass_reward"][rew_index] = self.strategic_pass_reward
                        reward[rew_index] += components["strategic_pass_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info

    def evaluate_pass_quality(self, observation):
        """
        Simplistic function to measure pass quality based on ball direction and player orientation
        Higher values signify better precision; arbitrary calculation as example.
        """
        ball_direction = np.linalg.norm(observation['ball_direction'])
        player_direction = np.linalg.norm(observation['left_team_direction'][observation['active']])
        return (ball_direction + player_direction) / 2
