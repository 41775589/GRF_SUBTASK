import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward system to focus on offensive strategies, team coordination,
    and adapting between scoring and positioning techniques."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_bonus = 0.2
        self.shooting_bonus = 0.5
        self.positioning_bonus = 0.1

    def reset(self):
        """Reset the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment to ensure continuity in serialized form."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from serialized form."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Compute custom reward based on improved offensive gameplay."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'passing_bonus': [0.0] * len(reward),
                      'shooting_bonus': [0.0] * len(reward),
                      'positioning_bonus': [0.0] * len(reward)}

        for index in range(len(reward)):
            player_obs = observation[index]
            ball_x, ball_y = player_obs['ball'][0], player_obs['ball'][1]
            
            # Add bonus for shots at the goal
            if player_obs['ball_owned_team'] == 0 and np.abs(ball_y) <= 0.044 and ball_x > 0.7:
                components['shooting_bonus'][index] = self.shooting_bonus

            # Add bonus for effective passing
            if 'sticky_actions' in player_obs and player_obs['sticky_actions'][-2] == 1:
                components['passing_bonus'][index] = self.passing_bonus

            # Positioning: Rewards player being in strategic locations
            if player_obs['right_team'][player_obs['active']][0] > 0:  # In opponent's half
                components['positioning_bonus'][index] = self.positioning_bonus

            # Combine rewards
            reward[index] += (components['shooting_bonus'][index] +
                              components['passing_bonus'][index] +
                              components['positioning_bonus'][index])

        return reward, components

    def step(self, action):
        """Step through environment, augment reward with components and add to info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
