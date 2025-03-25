import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering long passes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = np.zeros(3)
        self.owning_team_start = -1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = np.zeros(3)
        self.owning_team_start = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_position': self.last_ball_position,
            'owning_team_start': self.owning_team_start
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['CheckpointRewardWrapper']['last_ball_position']
        self.owning_team_start = from_pickle['CheckpointRewardWrapper']['owning_team_start']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_pass_reward": [0.0, 0.0]
        }
        
        # Update for valid observations only
        if observation is not None:
            for i in range(len(reward)):
                o = observation[i]
                ball_position = o['ball']

                if 'ball_owned_team' in o:
                    if o['ball_owned_team'] != self.owning_team_start:
                        # Change of possession reset the checkpoint
                        self.owning_team_start = o['ball_owned_team']
                        self.last_ball_position = ball_position

                    if o['ball_owned_team'] == 1 or o['ball_owned_team'] == 0:
                        distance = np.linalg.norm(ball_position - self.last_ball_position)
                        # Encourage longer passes with exponentially higher rewards
                        components["long_pass_reward"][i] = distance ** 2
                        reward[i] += components["long_pass_reward"][i]
                        # Update the last known position
                        self.last_ball_position = ball_position

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        rewards_and_components = [self.reward(reward_individual) for reward_individual in reward]
        reward, components = map(list, zip(*rewards_and_components))
        
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        self.sticky_actions_counter.fill(0)
        for agent_obs in observation:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
        
        return observation, reward, done, info
