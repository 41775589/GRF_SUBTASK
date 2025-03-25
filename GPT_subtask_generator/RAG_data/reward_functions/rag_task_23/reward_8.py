import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances coordination and role proficiency in defensive scenarios,
    specifically near the penalty area by providing rewards based on the players'
    position, active role, and ability to maintain advantageous defensive positions.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.penalty_area_threshold = 0.2  # Threshold to consider penalty area proximity
        self.right_team_roles = None

    def reset(self):
        """
        Resets the environment and the sticky actions counter.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the current state with custom data.
        """
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state with custom data.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Augments the default reward system by rewarding defensive coordination, 
        especially when players maintain positions in or near the penalty area in high-pressure scenarios.

        Parameters:
            reward (list[float]): List of base rewards from the environment.

        Returns:
            tuple[list[float], dict[str, list[float]]]: Adjusted rewards and the components dict of individual rewards.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "penalty_area_defense_reward": [0.0, 0.0]}

        for i, o in enumerate(observation):
            # Reward players based on their proximity to their own penalty area when defending
            player_pos = o['right_team'][o['active']] if o['ball_owned_team'] == 0 else o['left_team'][o['active']]
            if abs(player_pos[0]) > (1 - self.penalty_area_threshold):
                if o['ball_owned_player'] == -1 or o['ball_owned_team'] == 1:
                    # Increase reward for being in a defensive position near the penalty area without possession
                    components['penalty_area_defense_reward'][i] += 0.5

        # Combine the components to form the final reward list
        for i in range(len(reward)):
            reward[i] += components['penalty_area_defense_reward'][i]

        return reward, components

    def step(self, action):
        """
        Steps through the environment with an action and modifies the reward based on defined criteria.

        Parameters:
            action (var): Action taken by the agent.

        Returns:
            tuple: Observations, rewards, done flag, and additional info from the environment.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)

        # Add reward components to info for debugging purposes
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_val in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_val
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
