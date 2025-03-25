import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper enhancing man-marking, shot-blocking, and action stalling skills."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize the counter for sticky actions which are used for dribbling, sprinting, etc.
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Stopping coefficients
        self._block_shot_reward = 0.3
        self._man_marking_reward = 0.2
        self._possession_interrupt_reward = 0.1

    def reset(self):
        """ Reset the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Return the state that includes information from this wrapper."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state that includes information relevant to this wrapper."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """Modify the reward function by focusing on defensive skills."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'block_shot_reward': [0.0] * len(reward),
                      'man_marking_reward': [0.0] * len(reward),
                      'possession_interrupt_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            close_opponents = np.sqrt(np.sum((o['left_team'] - o['right_team'][o['active']], )**2, axis=1)) <= 0.015
            
            if o['game_mode'] == 3:  # Free kick game mode
                components['block_shot_reward'][rew_index] = self._block_shot_reward
                reward[rew_index] += components['block_shot_reward'][rew_index]
            
            # Man marking: reward for staying close to opponents.
            if o['ball_owned_team'] == 1 and np.any(close_opponents):
                components['man_marking_reward'][rew_index] = self._man_marking_reward
                reward[rew_index] += components['man_marking_reward'][rew_index]
            
            # Possession interrupt (lose possession or ball previously owned by this team)
            if o['ball_owned_team'] != 0 and 'ball_owned_team' in o:
                prev_ball_team = self.env.unwrapped.previous_observation()['ball_owned_team']
                if prev_ball_team == 0:
                    components['possession_interrupt_reward'][rew_index] = self._possession_interrupt_reward
                    reward[rew_index] += components['possession_interrupt_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Execute a step in the environment with the modified reward."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        # Aggregate final reward from all components
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Reset sticky actions counter
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
