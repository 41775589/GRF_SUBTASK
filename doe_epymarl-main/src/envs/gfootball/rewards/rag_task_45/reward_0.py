import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward for mastering abrupt stop-and-go techniques.
    Specifically, rewards effective defensive maneuvers using Stop-Sprint and Stop-Moving actions.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_positions = []
        self.previous_actions = []

    def reset(self):
        """Resets the environment and relevant variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_positions = []
        self.previous_actions = []
        return self.env.reset()

    def get_state(self, to_pickle):
        """Stores the state of the wrapper."""
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_positions': self.previous_positions,
            'previous_actions': self.previous_actions
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restores the state of the wrapper."""
        from_pickle = self.env.set_state(state)
        recovery_data = from_pickle['CheckpointRewardWrapper']
        self.previous_positions = recovery_data['previous_positions']
        self.previous_actions = recovery_data['previous_actions']
        return from_pickle

    def reward(self, reward):
        """
        Modifies the reward based on the game dynamics:
        Provides rewards for effective stop-and-go actions in defensive maneuvers.
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'stop_sprint_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for idx, obs in enumerate(observation):
            # basics - game mode normal
            if obs['game_mode'] != 0:
                continue

            player_pos = obs['right_team'][obs['active']]
            player_dir = obs['right_team_direction'][obs['active']]
            current_action = obs['sticky_actions']
            
            # identify Stop-Sprint and Stop-Moving sequences
            stop_action = football_action_set.action_idle
            if len(self.previous_actions) > idx:
                was_moving = np.any(self.previous_actions[idx][[0, 1, 3, 4, 6, 7]])  # any side movements
                just_stopped = current_action[stop_action] and was_moving
            
                if just_stopped:
                    # Reward stopping quickly after movement if facing opposing player closely
                    proximity_threshold = 0.1
                    for opponent in obs['left_team']:
                        distance = np.linalg.norm(player_pos - opponent)
                        if distance < proximity_threshold:
                            components['stop_sprint_reward'][idx] += 0.5
                            reward[idx] += components['stop_sprint_reward'][idx]
                            break

        self.previous_positions = [obs['right_team'][obs['active']] for obs in observation]
        self.previous_actions = [obs['sticky_actions'] for obs in observation]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for idx, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{idx}"] = action
        return observation, reward, done, info
