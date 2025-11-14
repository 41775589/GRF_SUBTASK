import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that emphasizes strategic defensive positioning and interaction among defenders."""

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
        return from_pickle

    def reward(self, reward):
        """Adjust reward based on defensive organization and pressure."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "positioning_reward": [0.0] * len(reward),
            "pressure_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for idx, obs in enumerate(observation):
            # Defensive positioning - reward players for being positioned in key defensive areas
            player_pos = obs['left_team'][obs['active']]
            player_role = obs['left_team_roles'][obs['active']]
            # Reward defenders for positioning closer to the goal area when opponents are near
            if player_role in [0, 1, 2, 3, 4] and player_pos[0] < -0.5:
                opponents_positions = obs['right_team']
                own_goal_distance = np.abs(player_pos[0] + 1)  # x-position normalized from -1 to 1
                closest_opponent_distance = np.min(np.linalg.norm(opponents_positions - player_pos, axis=1))
                # Increasing rewards as opponents get closer to the own goal area and player is closer to the goal
                components["positioning_reward"][idx] = 0.1 * (0.2 / (closest_opponent_distance + 0.1)) * (0.5 - own_goal_distance)

            # Defensive pressure - reward defenders for applying pressure on ball carrier
            if obs['ball_owned_team'] == 1:  # if the ball is with the opponent
                ball_pos = obs['ball'][:2]
                distance_to_ball = np.linalg.norm(player_pos - ball_pos)
                # Reward players for being close to the ball to apply pressure
                if distance_to_ball < 0.2:
                    components["pressure_reward"][idx] = 0.5 * (0.2 - distance_to_ball)
            
            # Update reward for this player by component contributions
            reward[idx] += components["positioning_reward"][idx] + components["pressure_reward"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        for agent_obs in observation:
            for i, action_val in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_val
                info[f"sticky_actions_{i}"] = action_val
        
        return observation, reward, done, info
