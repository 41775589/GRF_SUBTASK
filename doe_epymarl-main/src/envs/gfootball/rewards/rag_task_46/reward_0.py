import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward based on successful tackles and ball possession recovery without penalties."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.penalized_actions = 0
        self.successful_tackles = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.penalized_actions = 0
        self.successful_tackles = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        to_pickle['penalized_actions'] = self.penalized_actions
        to_pickle['successful_tackles'] = self.successful_tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        self.penalized_actions = from_pickle['penalized_actions']
        self.successful_tackles = from_pickle['successful_tackles']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        base_reward = reward.copy()
        component_rewards = {'base_score_reward': base_reward,
                             'tackle_reward': [0.0, 0.0],
                             'penalty_prevention_reward': [0.0, 0.0]}

        if observation is None:
            return reward

        for idx in range(len(reward)):
            o = observation[idx]
            ball_owned_team = o['ball_owned_team']
            has_ball = ball_owned_team == (0 if idx == 0 else 1)
            tackle_occurred = o['game_mode'] == 3  # FreeKick indicates foul or tackle

            if tackle_occurred and has_ball:
                self.successful_tackles += 1
                component_rewards['tackle_reward'][idx] = 1.0
            elif tackle_occurred and not has_ball:
                self.penalized_actions += 1
                component_rewards['penalty_prevention_reward'][idx] = -1.0

        # Compute final reward for this step
        reward = [(base_reward[idx] +
                 component_rewards['tackle_reward'][idx] +
                 component_rewards['penalty_prevention_reward'][idx])
                 for idx in range(len(reward))]
        return reward, component_rewards

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
