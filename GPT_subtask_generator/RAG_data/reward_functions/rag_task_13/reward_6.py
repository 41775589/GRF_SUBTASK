import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards focused on the defensive capabilities, particularly for the stopper role."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
    def reset(self):
        """Reset the environment and clear counters."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieve the current state for saving."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state based on loaded data."""
        return self.env.set_state(state)

    def reward(self, rewards):
        """Reward function focused on defensive actions by a stopper."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": rewards.copy(),
                      "block_reward": [0.0] * len(rewards),
                      "marking_reward": [0.0] * len(rewards)}

        if observation is None:
            return rewards, components
        
        assert len(rewards) == len(observation)

        for rew_index, o in enumerate(observation):
            # Reward for blocking shots
            if 'ball_direction' in o and 'right_team_direction' in o:
                ball_trajectory = o['ball_direction']
                player_trajectory = o['right_team_direction'][o['active']]
                # Calculate trajectories similarity, negative reward if ball is going towards own goal
                ball_to_goal_angle = np.linalg.norm(ball_trajectory - np.array([-1, 0]))
                player_to_ball_angle = np.linalg.norm(player_trajectory - ball_trajectory)

                if ball_to_goal_angle < 0.1 and player_to_ball_angle < 0.1:
                    components["block_reward"][rew_index] += 0.3
                    rewards[rew_index] += 0.3
            
            # Reward for marking opponent players closely
            if 'right_team' in o:
                opponent_distances = np.linalg.norm(o['right_team'] - o['right_team'][o['active']], axis=1)
                close_opponents = np.sum(opponent_distances < 0.1)
                components["marking_reward"][rew_index] += 0.1 * close_opponents
                rewards[rew_index] += 0.1 * close_opponents

        return rewards, components

    def step(self, action):
        """Override the default step function to include reward modification."""
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
