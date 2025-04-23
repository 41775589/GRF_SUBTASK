import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on ball control skills that facilitate transition from defense to attack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_bonus = 0.1
        self.continuous_control_bonus = 0.05
        self.ball_control_measures = {}
        self.previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_measures = {}
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.ball_control_measures
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_control_measures = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward

        components = {"base_score_reward": reward.copy(),
                      "pass_completion_bonus": [0.0] * len(reward),
                      "continuous_control_bonus": [0.0] * len(reward)}

        for idx, o in enumerate(observation):
            if o['ball_owned_player'] == o['active']:

                # Reward for ball possession
                if self.previous_ball_owner != o['ball_owned_player']:
                    components["continuous_control_bonus"][idx] = self.continuous_control_bonus
                    reward[idx] += components["continuous_control_bonus"][idx]

                # Reward for transition play (successful pass or long kick)
                long_play_action = self.sticky_actions_counter[4]  # assuming index 4 refers to a significant gameplay action such as a long kick
                if long_play_action:
                    components["pass_completion_bonus"][idx] = self.pass_completion_bonus
                    reward[idx] += components["pass_completion_bonus"][idx]

            self.previous_ball_owner = o['ball_owned_player']

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
