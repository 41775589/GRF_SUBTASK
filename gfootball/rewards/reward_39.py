import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive gameplay metrics: shooting accuracy, dribbling, and passing."""

    def __init__(self, env):
        super().__init__(env)
        # Initialize parameters for custom reward components
        self.shooting_accuracy_reward = 0.2
        self.effective_dribbling_reward = 0.15
        self.effective_passing_reward = 0.1
        # Tracking last positions to calculate movement and passing effectiveness
        self.last_positions = None

    def reset(self):
        # Reset the last positions when a new episode starts
        self.last_positions = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shooting_accuracy_reward'] = self.shooting_accuracy_reward
        to_pickle['effective_dribbling_reward'] = self.effective_dribbling_reward
        to_pickle['effective_passing_reward'] = self.effective_passing_reward
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shooting_accuracy_reward = from_pickle['shooting_accuracy_reward']
        self.effective_dribbling_reward = from_pickle['effective_dribbling_reward']
        self.effective_passing_reward = from_pickle['effective_passing_reward']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shooting_accuracy_reward": [0.0] * len(reward),
            "effective_dribbling_reward": [0.0] * len(reward),
            "effective_passing_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        current_positions = np.array([o['right_team'] for o in observation])

        if self.last_positions is not None:
            # Calculate movements
            movement_distances = np.linalg.norm(current_positions - self.last_positions, axis=-1)
            dribbling_effectiveness = movement_distances * components["effective_dribbling_reward"]

            # Reward for maintaining possession and advancing towards the opponent's goal
            for idx, o in enumerate(observation):
                if o['ball_owned_team'] == 1:  # Assuming controlled team is '1'
                    player_with_ball = o['ball_owned_player']
                    # Calculate effective passing reward if ball ownership changes to a teammate closer to the goal
                    if player_with_ball >= 0:  # Verify that a player has the ball
                        goal_distance_before = np.linalg.norm(self.last_positions[player_with_ball] - np.array([1, 0]))
                        goal_distance_after = np.linalg.norm(current_positions[player_with_ball] - np.array([1, 0]))
                        if goal_distance_after < goal_distance_before:
                            components["effective_passing_reward"][idx] += self.effective_passing_reward

                        # Add shooting accuracy reward if shot is taken and score increases
                        if o['game_mode'] == 6:  # 6 corresponds to a penalty for example
                            components["shooting_accuracy_reward"][idx] += self.shooting_accuracy_reward

                reward[idx] += 0.1 * dribbling_effectiveness[idx] + components["shooting_accuracy_reward"][idx] + components["effective_passing_reward"][idx]

        # Update last positions
        self.last_positions = current_positions

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Include the final and component rewards in the 'info' for diagnostics
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
