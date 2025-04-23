import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances training by focusing on shooting and passing with scenario-based rewards."""

    def __init__(self, env):
        super().__init__(env)
        self.shooting_distance_threshold = 0.8  # Threshold distance for considering shooting attempts
        self.passing_score = 0.2  # Reward for making a successful pass
        self.goal_score = 1.0  # Reward for scoring a goal
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Count of sticky actions for debugging purposes

    def reset(self):
        """Reset the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Custom reward function emphasizing passing and shooting."""
        new_rewards = []
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [],
                      "passing_reward": []}

        observations = self.env.unwrapped.observation()
        
        for i, r in enumerate(reward):
            o = observations[i]
            modified_reward = r

            # Check if a goal was scored
            if r == self.goal_score:
                components["shooting_reward"].append(self.goal_score)
                modified_reward += self.goal_score
            else:
                components["shooting_reward"].append(0)

            # Calculate the reward for passing
            if o['ball_owned_team'] == 0:
                if np.linalg.norm(o['ball_direction'][:2]) > self.shooting_distance_threshold:
                    # Check if the ball is moving towards the goal or teammates
                    goal_dir = np.sign(o['ball'][0])
                    ball_dir = np.sign(o['ball_direction'][0])
                    if goal_dir == ball_dir:
                        # Reward for shooting towards the goal
                        components["shooting_reward"][-1] = self.goal_score / 2
                        modified_reward += self.goal_score / 2
                    else:
                        # Reward for passing the ball
                        components["passing_reward"].append(self.passing_score)
                        modified_reward += self.passing_score
                else:
                    components["passing_reward"].append(0)
            else:
                components["passing_reward"].append(0)
            
            new_rewards.append(modified_reward)
        
        return new_rewards, components

    def step(self, action):
        """Perform a step using the specified action, adjust rewards and return observations."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                if action:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
