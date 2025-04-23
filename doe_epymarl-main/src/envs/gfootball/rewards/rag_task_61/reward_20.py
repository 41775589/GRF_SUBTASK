import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense situational reward for possession changes and strategic positioning."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize a dictionary to track player positions during key events
        self.previous_ball_position = None
        self.position_changes_tracker = {}

    def reset(self):
        # Reset the internal states for the new episode
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.position_changes_tracker = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_position': self.previous_ball_position,
            'position_changes_tracker': self.position_changes_tracker
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['CheckpointRewardWrapper']['previous_ball_position']
        self.position_changes_tracker = from_pickle['CheckpointRewardWrapper']['position_changes_tracker']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        if observation is None:
            return reward

        current_ball_position = observation[0]['ball'][:2]  # the x, y position of the ball

        for team in ['left_team', 'right_team']:
            # Iterate through observation to modify the rewards based on possession and strategic positioning
            for idx, player in enumerate(observation[0][team]):
                player_position = player[:2]  # x, y position of the player
                if self.previous_ball_position is not None:
                    # Calculate the movement of the ball towards this player since last step
                    distance_change = np.linalg.norm(current_ball_position - player_position) - np.linalg.norm(self.previous_ball_position - player_position)
                    if distance_change < 0:
                        # Ball has moved closer to this player
                        self.position_changes_tracker[team, idx] = self.position_changes_tracker.get((team, idx), 0) + 1
                        # Reward for moving towards ball possession
                        reward += 0.01 * self.position_changes_tracker[team, idx]

        if observation[0]['ball_owned_team'] != observation[1]['ball_owned_team']:
            # Possession change detected
            reward += 0.5  # Assign a larger reward for successful possession change

        self.previous_ball_position = current_ball_position.copy()

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)

        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
