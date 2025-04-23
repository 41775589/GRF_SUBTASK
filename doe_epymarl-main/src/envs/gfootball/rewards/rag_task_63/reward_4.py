import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that simulates goalkeeper training scenarios focusing on shot stopping, decision-making for ball distribution, and communication with defenders."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.goalkeeper_index = None
        self.ball_block_reward = 1.0
        self.distribution_reward = 0.5
        self.communication_reward = 0.2

    def reset(self):
        """Reset wrapped environment and reset auxiliary counters and state trackers."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.goalkeeper_index = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the internal state for serialization purposes."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        to_pickle['previous_ball_position'] = self.previous_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the internal state from deserialized state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        self.previous_ball_position = from_pickle.get('previous_ball_position', None)
        return from_pickle

    def reward(self, reward):
        """Modify reward based on goalkeeper training focused tasks."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "ball_block_reward": np.zeros_like(reward),
                      "distribution_reward": np.zeros_like(reward),
                      "communication_reward": np.zeros_like(reward)}

        if observation is None:
            return reward, components
        
        for idx in range(len(reward)):
            obs = observation[idx]
            if self.goalkeeper_index is None and 'right_team_roles' in obs:
                # Assume goalkeeper index is where role 0 (goalkeeper) is present
                self.goalkeeper_index = np.where(obs['right_team_roles'] == 0)[0][0]

            # Goalkeeper actions and positioning
            player_position = obs['right_team'][self.goalkeeper_index]
            ball_position = obs['ball'][:2]

            # Ball stopping reward calculation
            if self.previous_ball_position is not None:
                distance_to_goal = np.linalg.norm(ball_position - np.array([1, 0]))
                if obs['ball_owned_team'] == 1 and distance_to_goal < np.linalg.norm(self.previous_ball_position - np.array([1, 0])):
                    components['ball_block_reward'][idx] = self.ball_block_reward
                    reward[idx] += components['ball_block_reward'][idx]

            # Reward for distribution decisions
            if obs['game_mode'] == 6:  # Assume 6 is a game mode after catching the ball
                components['distribution_reward'][idx] = self.distribution_reward
                reward[idx] += components['distribution_reward'][idx]
            
            # Communication with defenders (simplified as maintaining positioning relative to closest defender)
            if 'right_team' in obs:
                defenders = np.delete(obs['right_team'], self.goalkeeper_index, axis=0)
                min_dist_to_defender = np.min(np.linalg.norm(defenders - player_position, axis=1))
                if min_dist_to_defender < 0.1:  # Assuming a threshold for good communication
                    components['communication_reward'][idx] = self.communication_reward
                    reward[idx] += components['communication_reward'][idx]

            self.previous_ball_position = ball_position.copy()

        return reward, components

    def step(self, action):
        """Performs a step in the underlying environment, adjusting rewards according to goalkeeper-specific enhancements."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
