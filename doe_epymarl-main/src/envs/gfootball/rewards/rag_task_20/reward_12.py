import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that manages offensive strategy optimizations."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.passing_threshold = 0.05
        self.shooting_range = 0.9
        self.position_bonus = 0.1
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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_bonus": [0.0] * len(reward),
                      "shooting_bonus": [0.0] * len(reward),
                      "position_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            obs = observation[idx]
            # Enhance reward for effective passing
            if obs['game_mode'] == 5: # assuming 5 is a play mode where a pass action is executed
                if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                    components["passing_bonus"][idx] = self.passing_threshold

            # Enhance reward for getting into shooting range
            if abs(obs['ball'][0]) > self.shooting_range:
                components["shooting_bonus"][idx] = np.exp(-abs(obs['ball'][1]))

            # Position bonus for players being forward in the field to increase aggressive gameplay
            if obs['ball_owned_team'] == 0:  # if own team has ball
                components["position_bonus"][idx] = self.position_bonus * obs['left_team'][obs['active']][0]

            # Aggregate the rewards modification
            reward[idx] = components["base_score_reward"][idx] + \
                          components["passing_bonus"][idx] + \
                          components["shooting_bonus"][idx] + \
                          components["position_bonus"][idx]

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
