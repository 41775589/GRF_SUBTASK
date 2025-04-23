import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for perfecting standing tackles during gameplay."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        # Access the last observation from the environment to determine if a tackle was successful.
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, obs in enumerate(observation):
            # Check both game mode (normal and free kicks) and if ball is owned by the opponent.
            if obs['game_mode'] in [0, 3] and obs['ball_owned_team'] == 1:  # opponent has the ball

                # Calculate distance to the ball and check active player
                ball_pos = np.array(obs['ball'][:2])  # ignore z-axis
                active_player_pos = obs['left_team'][obs['active']] if obs['ball_owned_team'] == 1 else obs['right_team'][obs['active']]
                distance_to_ball = np.linalg.norm(ball_pos - active_player_pos)

                # Reward for tackling (ball recovery within a small radius without causing a foul)
                if distance_to_ball < 0.015 and not obs['left_team_yellow_card'][obs['active']]:
                    components["tackle_reward"][index] = self.tackle_reward
                    reward[index] += components["tackle_reward"][index]

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
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
