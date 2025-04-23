import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds specific rewards for goalkeeper training."""

    def __init__(self, env):
        super().__init__(env)
        # Parameters for goalkeeper performance metrics
        self.goalkeeper_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_interactions = 0
        self.close_ball_intercepts = 0
        self.effective_clearances = 0

    def reset(self):
        """Resets the environment and metrics."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_interactions = 0
        self.close_ball_intercepts = 0
        self.effective_clearances = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the current state for serialization purposes."""
        to_pickle['goalkeeper_metrics'] = {
            'ball_interactions': self.ball_interactions,
            'close_ball_intercepts': self.close_ball_intercepts,
            'effective_clearances': self.effective_clearances
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore state from serialization."""
        from_pickle = self.env.set_state(state)
        metrics = from_pickle['goalkeeper_metrics']
        self.ball_interactions = metrics['ball_interactions']
        self.close_ball_intercepts = metrics['close_ball_intercepts']
        self.effective_clearances = metrics['effective_clearances']
        return from_pickle

    def reward(self, reward):
        """Calculate additional reward components based on goalkeeper performance."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "save_bonus": 0.0, "clearance_bonus": 0.0}
        if observation is None:
            return reward, components

        o = observation[0]  # Assuming goalkeeper is always the first player in the observation list

        ball_position = o['ball']
        player_position = o['left_team'][0]  # Assuming goalkeeper is the first player of left_team
        ball_owned_team = o['ball_owned_team']

        # Calculate distance to ball
        distance_to_ball = np.linalg.norm(np.array(player_position) - np.array(ball_position[:2]))

        # Reward for being close to ball when not owned and effectively clearing it
        if distance_to_ball < 0.1 and ball_owned_team != 0:
            self.effective_clearances += 1
            components["clearance_bonus"] = 1.0

        # Reward for saves and ball interceptions
        if 'ball_owned_player' in o and o['ball_owned_player'] == 0:
            self.ball_interactions += 1
            if distance_to_ball < 0.05:  # Close interaction with the ball
                self.close_ball_intercepts += 1
                components["save_bonus"] = 2.0

        # Calculate total reward
        final_rewards = [components['base_score_reward'][0] + components['save_bonus'] + components['clearance_bonus']]

        return final_rewards, components

    def step(self, action):
        """Execute environment step, adjusting reward based on goalkeeper actions."""
        observation, reward, done, info = self.env.step(action)
        adjusted_rewards, components = self.reward(reward)
        info.update({f"component_{key}": value for key, value in components.items()})
        info["final_reward"] = sum(adjusted_rewards)
        return observation, adjusted_rewards, done, info
