import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A specialized reward wrapper to encourage long accurate passes between different areas of the field.

    This wrapper provides additional rewards for successfully executing long passes that connect specific
    designated areas in the playfield, focusing on enhancing the agent's skill in vision, timing, and 
    precision in ball distribution.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Define several key areas on the field that are targets for long passes.
        self.pass_targets = [
            (-0.8, 0.0),  # Left side near mid-field
            (0.8, 0.0),   # Right side near mid-field
            (0.0, 0.4),   # Center close to opponent's goal
            (0.0, -0.4),  # Center close to own goal
        ]
        self.target_radius = 0.1  # Tolerance radius around each target
        self.recent_passes = []  # Tracks recent successful passes
        self.pass_reward = 0.2   # Reward for each successful passing
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the reward wrapper state for a new episode."""
        self.recent_passes = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Enhance the reward based on the successful execution of long passes.

        Args:
            reward (list[float]): original reward from the base environment.

        Returns:
            (list[float], dict[str, list[float]]): Tuple of modified rewards and component breakdown.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "pass_completion_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for i in range(len(observation)):
            o = observation[i]
            ball_owner_team = o['ball_owned_team']
            last_action_successful = len(self.recent_passes) > 0 and self.recent_passes[-1][0] != ball_owner_team

            if ball_owner_team == 1 and 'ball_owned_player' in o and o['ball_owned_player'] != -1:
                ball_pos = o['ball'][:2]

                if last_action_successful:
                    start_pos, end_pos = self.recent_passes[-1][1:]  # unpack last pass start and end positions
                    
                    # Check if the pass target has been achieved
                    for target in self.pass_targets:
                        dist = np.sqrt((end_pos[0] - target[0]) ** 2 + (end_pos[1] - target[1]) ** 2)
                        if dist <= self.target_radius:
                            components["pass_completion_reward"][i] += self.pass_reward
                            reward[i] += components["pass_completion_reward"][i]

                # Update recent passes tracking
                if len(self.recent_passes) == 0 or self.recent_passes[-1][0] != ball_owner_team:
                    self.recent_passes.append((ball_owner_team, start_pos, ball_pos))
                else:
                    self.recent_passes[-1] = (ball_owner_team, start_pos, ball_pos)

        return reward, components

    def step(self, action):
        """
        Step through the environment with the given actions and apply reward modifications.

        Args:
            action: action to take in the environment.

        Returns:
            Tuple representing the result of the environment step followed by modified rewards.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_val in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_val
        return observation, reward, done, info
