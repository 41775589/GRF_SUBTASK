import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    This reward wrapper focuses on enhancing the team's effectiveness in mid to long-range passing.
    It rewards precision and strategic use of high and long passes:
    - Reward for successful long-range passes.
    - Reward for maintaining possession after a long-range pass.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_reward = 0.05
        self.possession_reward = 0.03
        # Store data about last ball position and ownership to detect successful long passes
        self.last_ball_position = None
        self.last_ball_owned_team = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.last_ball_owned_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self.last_ball_position
        to_pickle['last_ball_owned_team'] = self.last_ball_owned_team
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle['last_ball_position']
        self.last_ball_owned_team = from_pickle['last_ball_owned_team']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward),
                      "possession_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            curr_observation = observation[i]
            current_ball_position = curr_observation['ball'][:2]  # Only X, Y positions

            if self.last_ball_position is not None and self.last_ball_owned_team is not None:
                if curr_observation['ball_owned_team'] == self.last_ball_owned_team:
                    distance_moved = np.linalg.norm(np.array(current_ball_position) - np.array(self.last_ball_position))
                    # Check if it was a long pass
                    if distance_moved > 0.3:  # Threshold for considering a pass long-range
                        components['long_pass_reward'][i] = self.pass_reward
                        reward[i] += self.pass_reward

                    # Reward for keeping possession after a long pass
                    if distance_moved > 0.3 and curr_observation['ball_owned_team'] == self.env.unwrapped.controlled_players()[0]:
                        components['possession_reward'][i] = self.possession_reward
                        reward[i] += self.possession_reward

            self.last_ball_position = current_ball_position
            self.last_ball_owned_team = curr_observation['ball_owned_team']

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
