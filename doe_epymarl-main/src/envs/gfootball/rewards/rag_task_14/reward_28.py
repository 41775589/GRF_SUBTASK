import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for 'sweeper' actions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the sticky actions counter and other required variables."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Retrieve the internal state required for pickling."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state when loading from pickled data."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Calculate rewards based on 'sweeper' activities."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'clearance_reward': [0.0] * len(reward),
                      'last_man_tackle_reward': [0.0] * len(reward),
                      'position_support_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Reward for clearing the ball when close to own goal (within 0.1 of x=-1)
            if o['ball'][0] < -0.9:
                if o['ball_owned_team'] == 0:  # Left team controls the ball
                    if o['left_team_roles'][o['ball_owned_player']] in {1, 2, 3, 4}:  # Defensive players
                        components['clearance_reward'][rew_index] = 0.3
                        reward[rew_index] += components['clearance_reward'][rew_index]

            # Reward for last-man tackles
            if o['game_mode'] in {3, 5}:  # Free kick or throw-in situations
                if o['ball_owned_team'] == 1:  # Right team controls the ball
                    if o['ball'][0] < -0.3 and np.abs(o['ball'][1]) < 0.2:
                        components['last_man_tackle_reward'][rew_index] = 0.5
                        reward[rew_index] += components['last_man_tackle_reward'][rew_index]

            # Reward for maintaining strategic positioning for support
            components['position_support_reward'][rew_index] = -0.01 * np.abs(o['ball'][0] - o['left_team'][o['active']][0])
            reward[rew_index] += components['position_support_reward'][rew_index]

        return reward, components

    def step(self, action):
        """Compute the step updating reward components and sticky actions."""
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
