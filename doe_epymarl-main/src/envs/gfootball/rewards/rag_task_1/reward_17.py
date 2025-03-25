import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dynamic reward for aggressive quick attack tactics during varied game phases."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.quick_attack_bonus = 0.05
        self.defensive_position_penalty = 0.01
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """Enhance reward if the agent performs quick attacking actions during varied game modes."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "quick_attack_bonus": [0.0] * len(reward),
                      "position_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, obs in enumerate(observation):
            base_reward = reward[index]
            original_position = obs['right_team'][obs['active']]
            ball_position = obs['ball'][:2]  # Get x, y positions only
            game_mode = obs['game_mode']

            dist_to_ball = np.linalg.norm(np.array(original_position) - np.array(ball_position))

            # Provide extra reward for staying close to the ball (indicative of offensive action)
            if dist_to_ball < 0.2 and obs['ball_owned_team'] == 1 and game_mode in [0, 2, 3, 4]:
                components['quick_attack_bonus'][index] = self.quick_attack_bonus
                reward[index] += components['quick_attack_bonus'][index]

            # Apply penalties if in a defensive position during active attack phases (encourage forward movement)
            if original_position[0] < 0 and obs['game_mode'] == 0 and ball_position[0] > original_position[0]:
                components['position_penalty'][index] = -self.defensive_position_penalty
                reward[index] += components['position_penalty'][index]

            # Normalize reward modification
            reward[index] = base_reward + components['quick_attack_bonus'][index] + components['position_penalty'][index]

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
            for i, action_presence in enumerate(agent_obs['sticky_actions']):
                if action_presence:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = action_presence
        return observation, reward, done, info
