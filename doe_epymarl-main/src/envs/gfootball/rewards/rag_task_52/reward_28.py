import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward focusing on defending strategies."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.tackle_threshold = 0.05  # Threshold distance to consider 'near' to opponent
        self.move_efficiency_reward = 0.1  # Reward for effective movement
        self.tackle_reward = 0.2  # Reward for getting close to opponent with the ball
        self.pressure_pass_reward = 0.15  # Reward for passing under pressure

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            "sticky_actions_counter": self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "move_efficiency_reward": [0.0] * len(reward),
                      "pressure_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Calculate distance to nearest opponent when the ball is owned by the opponent
            if o['ball_owned_team'] == 1:  # Assuming '0' is our team, '1' is opponent
                distances = np.linalg.norm(o['left_team'] - o['right_team'][o['ball_owned_player']], axis=1)
                # Check if any player is close enough to tackle
                if np.any(distances < self.tackle_threshold):
                    reward[rew_index] += self.tackle_reward
                    components["tackle_reward"][rew_index] = self.tackle_reward

            # Reward for controlled movement: no excessive movement when not near ball
            if o['ball_owned_team'] != 0 or (o['ball_owned_team'] == 0 and o['ball_owned_player'] != rew_index):
                if all(action == 0 for action in o['sticky_actions'][0:7]):  # Check if there are no movement actions
                    reward[rew_index] += self.move_efficiency_reward
                    components["move_efficiency_reward"][rew_index] = self.move_efficiency_reward

            # Reward for performing a pass under pressure
            if o['sticky_actions'][9]:  # Assuming action '9' corresponds to passing
                if o['ball_owned_team'] == 0 and np.any(np.linalg.norm(o['left_team'] - o['ball'], axis=1) < self.tackle_threshold):
                    reward[rew_index] += self.pressure_pass_reward
                    components["pressure_pass_reward"][rew_index] = self.pressure_pass_reward

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
