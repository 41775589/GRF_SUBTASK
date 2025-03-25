import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances central midfield gameplay focusing on controlled transitions and pace."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.05
        self._collected_checkpoints = {}

    def reset(self):
        """Reset all counters and the environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._collected_checkpoints = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include wrapper's state in the pickle state."""
        to_pickle['CheckpointRewardWrapper'] = self._collected_checkpoints
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set wrapper's state from the pickle state."""
        from_pickle = self.env.set_state(state)
        self._collected_checkpoints = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """Calculate and return enhanced reward based on midfield control and pacing."""
        
        observation = self.env.unwrapped.observation()  # Get raw observations from the environment
        components = {
            "base_score_reward": reward.copy(), 
            "transition_reward": [0.0] * len(reward),
            "pace_control_reward": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            self.handle_transitions(o, i, reward, components)

        return reward, components

    def handle_transitions(self, o, idx, reward, components):
        """Manage transition and pace rewards based on game state and player positions."""
        if o['game_mode'] != 0:  # Only apply in normal game mode
            return

        # Reward for maintaining ball control with central midfielders
        if o['ball_owned_team'] == 0 and o['left_team_roles'][o['active']] == 5:
            d_ball_goal = np.linalg.norm(o['ball'][:2] - [1, 0])  # Distance from ball to goal
            checkpoints_collected = self._collected_checkpoints.get(idx, 0)
            if checkpoints_collected < self._num_checkpoints and d_ball_goal < 0.2 * (5 - checkpoints_collected):
                components['transition_reward'][idx] = self._checkpoint_reward
                reward[idx] += components['transition_reward'][idx]
                self._collected_checkpoints[idx] = checkpoints_collected + 1

        # Reward for pace control
        v_player = np.linalg.norm(o['left_team_direction'][o['active']])  # Velocity magnitude of the controlled player
        if v_player < 0.01:  # Encourage less rushing, more strategic movement
            components['pace_control_reward'][idx] = 0.1
            reward[idx] += components['pace_control_reward'][idx]

    def step(self, action):
        """Step through the environment and modify the rewards."""
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
