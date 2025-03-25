import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes excellent defensive plays, specifically sliding tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(), 
                      "defensive_reward": np.zeros_like(reward)}

        for i, obs in enumerate(observation):
            if obs['game_mode'] not in [0]:  # Only normal play is considered
                continue

            # Detect situations where a slide tackle would be effective:
            # The agent must be very close to an opponent or the ball
            proximity_threshold = 0.05  # Distance threshold to consider a defensive action
            ball_pos = obs['ball'][:2]
            player_pos = obs[obs['controlled_player_side'] + '_team'][obs['active']]

            distance_to_ball = np.linalg.norm(player_pos - ball_pos)

            # Reward for close proximity to ball indicating potential defensive positioning
            if distance_to_ball < proximity_threshold:
                components['defensive_reward'][i] += 0.1

            # Additional reward for executing a slide action successfully while near the ball or an opponent
            if obs['sticky_actions'][7] == 1:  # 'action_slide'
                if distance_to_ball < proximity_threshold:
                    components['defensive_reward'][i] += 0.5

        reward += components['defensive_reward']
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
