import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized goalkeeper coordination reward."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_backup_reward = 0.1
        self.clear_ball_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_backup_reward": [0.0] * len(reward),
                      "clear_ball_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the active player is the goalkeeper.
            if o['active'] != 0:  # Assuming index 0 is the goalkeeper
                continue

            # Reward for being close to the ball when the game mode is high-pressure
            if o['game_mode'] in {2, 3, 4, 5, 6}:  # Representing high-pressure modes
                distance_to_ball = np.linalg.norm(o['left_team'][0] - o['ball'][:2])
                if distance_to_ball < 0.1:  # threshold for being 'close' to the ball
                    components["goalkeeper_backup_reward"][rew_index] = self.goalkeeper_backup_reward
                    reward[rew_index] += components["goalkeeper_backup_reward"][rew_index]

            # Reward for clearing the ball effectively to teammates not under pressure
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == 0:  # Goalkeeper has the ball
                # Find teammates not surrounded by opponents
                for player_idx, player_pos in enumerate(o['left_team']):
                    if np.any(np.linalg.norm(o['right_team'] - player_pos, axis=1) < 0.2):
                        continue  # Teammate is marked by an opponent
                    # Assume a successful pass is a clear
                    components["clear_ball_reward"][rew_index] = self.clear_ball_reward
                    reward[rew_index] += components["clear_ball_reward"][rew_index]
                    break

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
