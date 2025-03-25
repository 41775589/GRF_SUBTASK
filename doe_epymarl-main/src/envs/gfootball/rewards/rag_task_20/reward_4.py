import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward based on offensive play involving passing, positioning, and shooting."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        self.previous_ball_position = np.zeros(3)
        self.shooting_reward = 1.0
        self.passing_reward = 0.5
        self.positioning_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = np.array([0.0, 0.0, 0.0])
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.previous_ball_position.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, {"base_score_reward": reward.copy()}

        assert len(reward) == len(observation)

        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        for rew_index, o in enumerate(observation):
            ball_position = o['ball']
            ball_distance = np.linalg.norm(ball_position[:2] - np.array([-1, 0]))  # Distance to opponent's goal

            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:  # Assuming team 0 is the agent's team
                if np.linalg.norm(ball_position - self.previous_ball_position) > 0.1:
                    reward[rew_index] += self.passing_reward
                    components["passing_reward"][rew_index] = self.passing_reward

                # Closer positioning to opponent's goal
                previous_distance = np.linalg.norm(self.previous_ball_position[:2] - np.array([-1, 0]))
                if ball_distance < previous_distance:
                    reward[rew_index] += self.positioning_reward
                    components["positioning_reward"][rew_index] = self.positioning_reward

                # Check if a goal was scored
                if o['score'][0] > o['score'][1]:  # Assuming scoring increases the first element
                    components["shooting_reward"][rew_index] = self.shooting_reward
                    reward[rew_index] += self.shooting_reward

            # Update previous ball position for next reward calculation
            self.previous_ball_position = ball_position

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
