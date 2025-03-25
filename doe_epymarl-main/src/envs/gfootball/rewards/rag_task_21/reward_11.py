import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for defensive actions including interceptions and maintaining proper positioning."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interception_reward = 0.5
        self.positioning_reward = 0.3
        self.interceptions_completed = set()
        self.good_position_count = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.interceptions_completed = set()
        self.good_position_count = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        state_data = {'interceptions_completed': self.interceptions_completed,
                      'good_position_count': self.good_position_count}
        to_pickle['CheckpointRewardWrapper'] = state_data
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.interceptions_completed = state_data['interceptions_completed']
        self.good_position_count = state_data['good_position_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()  # Assuming access to observations
        components = {
            "base_score_reward": reward.copy(),
            "interception_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components
        
        for i, o in enumerate(observation):
            # Reward for intercepting the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                if o['active'] not in self.interceptions_completed:
                    components["interception_reward"][i] = self.interception_reward
                    reward[i] += components["interception_reward"][i]
                    self.interceptions_completed.add(o['active'])
            
            # Reward for maintaining good defensive positioning
            my_pos = o['left_team'][o['active']]
            ball_pos = o['ball']
            distance_to_ball = np.linalg.norm(my_pos[:2] - ball_pos[:2])

            # Assuming a good position is being close enough to potentially intercept or challenge
            if distance_to_ball < 0.3:
                self.good_position_count[o['active']] = self.good_position_count.get(o['active'], 0) + 1
                if self.good_position_count[o['active']] > 5:
                    components["positioning_reward"][i] = self.positioning_reward
                    reward[i] += components["positioning_reward"][i]
        
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
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += act
        return observation, reward, done, info
