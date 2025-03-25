import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for enhancing shooting skills."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_power_threshold = 0.5  # Define a threshold for considering it a powerful shot
        self.shot_accuracy_factor = 0.9  # Minimum factor of ball direction alignment to the goal for accuracy

    def reset(self):
        """Reset the sticky actions counter and environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """State getter that includes checkpoint data for consistency."""
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """State setter that ensures checkpoints reward states are correctly restored."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Custom reward function focused on shooting skills."""
        observation = self.env.unwrapped.observation()
        
        components = {
            "base_score_reward": reward.copy(),
            "shot_power_reward": np.zeros(len(reward)),
            "shot_accuracy_reward": np.zeros(len(reward))
        }

        if len(observation) == 0:
            return reward, components

        for i in range(len(reward)):
            # Check if the shot was made
            if self.env.unwrapped.observation()[i]['sticky_actions'][9] == 1: # Assuming shot action is the 10th in index.
                ball_speed = np.linalg.norm(observation[i]['ball_direction'][:2])
                
                # Reward powerful shots
                if ball_speed > self.shot_power_threshold:
                    components['shot_power_reward'][i] = 0.1 * ball_speed
                
                # Reward accuracy (alignment of shot vector towards the goal)
                ball_direction = observation[i]['ball_direction'][:2]
                goal_direction = np.array([1, 0]) if observation[i]['ball_owned_team'] == 0 else np.array([-1, 0])
                
                accuracy = np.dot(ball_direction, goal_direction) / (np.linalg.norm(ball_direction) * np.linalg.norm(goal_direction))
                if accuracy > self.shot_accuracy_factor:
                    components['shot_accuracy_reward'][i] = 0.2 * accuracy
            
            # Sum up the components
            reward[i] += components['shot_power_reward'][i] + components['shot_accuracy_reward'][i]

        return reward, components

    def step(self, action):
        """Step the environment and apply the new reward function."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
