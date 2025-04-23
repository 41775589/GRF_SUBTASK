import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies the reward based on strategic positioning and possession changes,
    aiming to improve team synergy during possession turnover and positioning during offensive 
    and defensive plays.
    """
    def __init__(self, env):
        super().__init__(env)
        # Initialize variables for tracking changes in possession and player positions
        self.last_ball_owned_team = None
        # Actions associated with losing or gaining control are given added emphasis through rewards
        self.possession_change_reward = 0.5
        self.positioning_reward_scale = 0.1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_owned_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_owned_team': self.last_ball_owned_team,
            'sticky_actions_counter': self.sticky_actions_counter.tolist()
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'CheckpointRewardWrapper' in from_pickle:
            self.last_ball_owned_team = from_pickle['CheckpointRewardWrapper']['last_ball_owned_team']
            self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_change_reward": 0.0,
                      "positioning_reward": 0.0}
        
        if observation is None:
            return reward, components

        current_ball_owned_team = observation['ball_owned_team']
        # Check for possession change
        if self.last_ball_owned_team is not None and current_ball_owned_team != self.last_ball_owned_team:
            reward += self.possession_change_reward
            components["possession_change_reward"] = self.possession_change_reward
        
        # Update last known possession
        self.last_ball_owned_team = current_ball_owned_team

        # Calculate positioning reward based on proximity to strategic locations
        ball_position = np.array(observation['ball'][:2])
        goal_position = np.array([1, 0]) if current_ball_owned_team == 1 else np.array([-1, 0])
        # Reward players for positioning the ball close to the opponent's goal
        distance_to_goal = np.linalg.norm(ball_position - goal_position)
        components["positioning_reward"] = (1 - distance_to_goal) * self.positioning_reward_scale
        reward += components["positioning_reward"]

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
