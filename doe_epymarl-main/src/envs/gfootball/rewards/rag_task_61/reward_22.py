import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic reward based on team synergy and possession changes."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_change_bonus = 0.2  # Bonus reward for successful possession change
        self.team_position_factor = 0.1  # Impact of good positioning on the reward

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}  # State saving logic if necessary
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # State setting logic if necessary
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'possession_change_bonus': [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            initial_reward = reward[rew_index]
            ball_change_reward = 0
            team_synergy_reward = 0

            # Bonus for changing possession from opponents to us
            if o['ball_owned_team'] == 0 and self.prev_ball_owned_team == 1:
                ball_change_reward = self.possession_change_bonus

            # Encourage strategic positioning based on team roles and positions
            if o['ball_owned_team'] == 0:  # Our team has the ball
                team_positions = o['left_team']
                ball_position = o['ball'][:2]
                distances = np.linalg.norm(team_positions - ball_position, axis=1)
                team_synergy_reward = -np.mean(distances) * self.team_position_factor

            reward[rew_index] = initial_reward + ball_change_reward + team_synergy_reward
            components['possession_change_bonus'][rew_index] = ball_change_reward + team_synergy_reward
        
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
