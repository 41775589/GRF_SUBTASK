import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive adaptation reward for precise stopping and starting movements."""

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
        components = {"base_score_reward": reward.copy(), "defensive_adaptation_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
            
        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            ball_pos = o['ball']
            sticky_actions = o['sticky_actions']
            player_pos = o['left_team' if o['ball_owned_team'] == 0 else 'right_team'][o['active']]
            distance_to_ball = np.linalg.norm(ball_pos[:2] - player_pos)
            
            # Encourage players to be ready to stop/start around the ball
            if distance_to_ball < 0.1 and (sticky_actions[8] == 0 or sticky_actions[9] == 0):  # not sprinting nor dribbling
                components["defensive_adaptation_reward"][rew_index] = 0.2  # reward for precise positioning without high motion
            
            # Apply accumulated reward components
            reward[rew_index] += components["defensive_adaptation_reward"][rew_index]

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
