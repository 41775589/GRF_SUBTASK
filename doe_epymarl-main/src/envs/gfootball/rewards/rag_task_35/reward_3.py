import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that provides rewards based on maintaining strategic positioning
    and ensuring effective transitions between defensive and offensive play."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = np.zeros(3)  # Initial ball position (x, y, z)
        self.last_positions = []
        self.strategic_positions = {
            'defensive': np.array([0, -0.2]),  # Closer to own goal
            'offensive': np.array([0, 0.2])   # Closer to opponent's goal
        }
        self.position_rewards = {
            'defensive': 0.05,
            'offensive': 0.05
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_positions = []
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['last_ball_position'] = self.last_ball_position
        state['last_positions'] = self.last_positions
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['last_ball_position']
        self.last_positions = from_pickle['last_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        new_reward = reward.copy()
        components = {
            "base_score_reward": reward.copy(),
            "strategic_position_reward": [0.0] * len(reward)
        }

        if observation is None:
            return new_reward, components

        for rew_index, o in enumerate(observation):
            player_pos = o['right_team' if o['ball_owned_team'] == 1 else 'left_team'][o['active']]
            ball_pos = o['ball'][:2]  # Ignore z dimension

            # Update ball position tracking
            if np.linalg.norm(ball_pos - self.last_ball_position) > 0.01:  # Ball has moved significantly
                self.last_ball_position = ball_pos.copy()

            # Check strategic position change rewards
            for key, position in self.strategic_positions.items():
                pos_dif = np.linalg.norm(player_pos - position)
                if pos_dif < 0.1:  # Close to strategic position
                    if len(self.last_positions) > rew_index:
                        last_pos = self.last_positions[rew_index]
                        if np.linalg.norm(last_pos - player_pos) > 0.05:  # Player has moved
                            components["strategic_position_reward"][rew_index] += self.position_rewards[key]
                            new_reward[rew_index] += self.position_rewards[key]

            # Update last position track
            if len(self.last_positions) > rew_index:
                self.last_positions[rew_index] = player_pos
            else:
                self.last_positions.append(player_pos)

        return new_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
