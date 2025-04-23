import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for optimizing shooting angles and timing under high-pressure scenarios near the goal."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goal_area_threshold = 0.6  # Approximate goal area threshold on x-axis

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
        components = {"base_score_reward": reward.copy(), "shooting_angle_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball'][0]  # Only consider x position for simplicity

            # Focus on high-pressure shooting scenarios
            if ball_pos > self.goal_area_threshold:
                # Check if the active team owns the ball near the opponent goal
                if o['ball_owned_team'] == 1 or (o['ball_owned_team'] == 0 and abs(o['ball'][1]) < 0.1):
                    # Reward calculated based on proximity to centre goal line on y-axis and ball control
                    components['shooting_angle_reward'][rew_index] = 0.2 * (1 - abs(o['ball'][1]))
                    reward[rew_index] += components['shooting_angle_reward'][rew_index]

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
                self.sticky_actions_counter[i] += action
        
        return observation, reward, done, info
