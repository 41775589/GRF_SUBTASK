import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for offensive capabilities focusing on fast-paced maneuvers and
    precision-based finishing. The reward is given based on proximity to the goal, control of the ball,
    and quick direction changes near the opponent's goal area.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the sticky actions counter and environment at the start of each new episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the environment including the sticky actions data.
        """
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        """
        Set the state of the environment from a pickle, including the sticky actions data.
        """
        env_data = self.env.set_state(from_pickle)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter')
        return env_data

    def reward(self, reward):
        """
        Enhance the reward function to focus on fast-paced offensive maneuvers:
        - Rewards for moving towards the opponent's goal with the ball.
        - Penalty for losing ball possession in critical areas.
        - Bonus for direction changes near the opponent's goal to simulate evasion capabilities.
        """
        observation = self.env.unwrapped.observation()
        if not observation:
            return reward

        components = {
            "base_score_reward": reward,
            "possession_reward": np.zeros_like(reward),
            "goal_proximity_reward": np.zeros_like(reward),
            "maneuver_reward": np.zeros_like(reward)
        }

        for idx, obs in enumerate(observation):
            ball_owned_team = obs['ball_owned_team']
            goal_x = 1 if ball_owned_team == 1 else -1  # Considering right goal is +1 and left goal is -1 on the x-axis

            # Check if the player's team owns the ball
            if ball_owned_team == 1:
                player_control = obs['active']
                ball_distance_to_goal = abs(obs['ball'][0] - goal_x)  # Distance from ball to goal on x-axis only

                # Reward for having possession and moving towards the goal
                if obs['ball_owned_player'] == player_control:
                    components['possession_reward'][idx] = 0.1
                    components['goal_proximity_reward'][idx] = (1 - ball_distance_to_goal) * 0.2

                # Additional reward for maneuverability within the goal area in the offensive phase
                if ball_distance_to_goal < 0.2:
                    components['maneuver_reward'][idx] = np.sum(obs['sticky_actions'][4:6]) * 0.05  # right and left movements

            # Compile the components into the total reward
            total_modified_reward = (reward[idx] + sum(component[idx] for component in components.values()))

            # Verify the reward with boundary conditions
            reward[idx] = np.clip(total_modified_reward, -1, 1)

        return reward, components

    def step(self, action):
        """
        Collect observations, apply action, recompute rewards and return info with reward breakdown.
        """
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()

        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return obs, reward, done, info
