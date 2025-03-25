import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards focusing on offensive strategies and team coordination."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_bonus = 0.1
        self.scoring_bonus = 1.0
        self.possession_bonus = 0.05
        self.positioning_bonus = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "passing_bonus": [0.0] * len(reward), 
                      "scoring_bonus": [0.0] * len(reward), 
                      "possession_bonus": [0.0] * len(reward),
                      "positioning_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            # Reward passing behaviors
            # Determine by increase in ball possession stats
            if o['ball_owned_team'] == 0 and 'ball_owned_player' in o:
                components['possession_bonus'][i] = self.possession_bonus
                reward[i] += components['possession_bonus'][i]

            # Reward scoring
            right_goal_x = 1
            if abs(o['ball'][0] - right_goal_x) < 0.1:
                components['scoring_bonus'][i] = self.scoring_bonus
                reward[i] += components['scoring_bonus'][i]

            # Reward positioning near the opponent goal
            distance_to_goal = np.sqrt((o['ball'][0] - right_goal_x)**2 + o['ball'][1]**2)
            if distance_to_goal < 0.3:
                components['positioning_bonus'][i] = self.positioning_bonus
                reward[i] += components['positioning_bonus'][i]

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
