import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances shot precision skills specifically for scenarios within close range of the goal,
       including angles and power adjustment required to beat the goalkeeper from a tight space."""

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
        # State restoration if needed
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "precision_reward": [0.0, 0.0]}  # As two agents are typically used

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_pos = o['ball']
            player_pos = o['left_team'][o['active']]
            goal_pos = [1, 0]  # Assuming playing from left to right

            # Calculate the distance to the goal and the angle of shot
            goal_dist = np.linalg.norm(np.subtract(goal_pos, player_pos))
            ball_to_goal_dist = np.linalg.norm(np.subtract(goal_pos, ball_pos))
            angle_to_goal = np.arccos((ball_to_goal_dist**2 + goal_dist**2 - np.linalg.norm(np.subtract(ball_pos, player_pos))**2) / (2 * ball_to_goal_dist * goal_dist))

            # Encourage shots close to goal & with a tight angle
            if goal_dist < 0.3 and abs(angle_to_goal) < np.pi / 4:
                components['precision_reward'][rew_index] = 0.5  # Reward for taking a precise shot within a good position
                reward[rew_index] += components['precision_reward'][rew_index]
        
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
