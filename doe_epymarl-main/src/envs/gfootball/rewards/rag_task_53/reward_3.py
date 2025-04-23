import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focused on ball control under pressure and strategic play."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_rewards = {}
        self.pressure_threshold = 0.5  # distance to consider under pressure
        self.ball_control_increment = 0.05  # reward increment for controlled plays

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_rewards = {}
        return self.env.reset()
        
    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_ball_control_rewards'] = self.ball_control_rewards
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_control_rewards = from_pickle['CheckpointRewardWrapper_ball_control_rewards']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "ball_control_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_under_pressure = self.calculate_pressure(o['left_team'], o['right_team'], o['ball'])
            if o['ball_owned_team'] == 0 and is_under_pressure:  # ball owned by left team and under pressure
                self.ball_control_rewards[rew_index] = self.ball_control_rewards.get(rew_index, 0) + self.ball_control_increment
                components["ball_control_reward"][rew_index] = self.ball_control_rewards[rew_index]

            reward[rew_index] += components["ball_control_reward"][rew_index]
        
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

    def calculate_pressure(self, left_team, right_team, ball):
        """Calculate if the ball is under pressure by opponents based on the proximity."""
        own_team_pos = np.array(left_team)
        opposition_pos = np.array(right_team)
        ball_pos = np.array(ball[:2])  # only x, y coordinates
        distances = np.linalg.norm(opposition_pos - ball_pos, axis=1)
        # Check if any opponent is within the pressure threshold distance
        return np.any(distances < self.pressure_threshold)
