import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for shooting precision and power in challenging scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shooting_practice_points = 0

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.shooting_practice_points = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        to_pickle['ShootingPoints'] = self.shooting_practice_points
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        self.shooting_practice_points = from_pickle['ShootingPoints']
        return from_pickle

    def reward(self, reward):
        # Access the observation directly to compute additional rewards
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        if observation is None:
            return reward, components
        
        components["shooting_practice"] = [0.0]

        # Assuming single agent case here
        o = observation[0]
        ball_pos = o['ball'][:2]
        # Distance to the opponent's goal
        distance_to_goal = np.abs(ball_pos[0] - 1)

        # Check if the ball is owned by the agent's team and near the opponent's goal
        if o['ball_owned_team'] == 1 and ball_pos[0] > 0.6:
            # Closer to the goal gives higher additional reward
            goal_proximity_reward = (1 - distance_to_goal) * 0.3
            components["shooting_practice"][0] += goal_proximity_reward

            # Further incentivize goal shooting action
            if 'action' in o and o['action'] == 'shot':
                components["shooting_practice"][0] += 0.5

        # Update the base reward with the dense components added
        reward[0] += components["shooting_practice"][0]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
