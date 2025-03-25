import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds dense rewards for mastering long passes.
    This wrapper focuses on rewarding the agents based on how effectively they perform long passes
    under various match conditions, taking distance and accuracy into consideration.
    """

    def __init__(self, env):
        super().__init__(env)
        self.pass_threshold = 0.5  # Threshold to consider a pass as "long"
        self.accuracy_bonus = 0.2  # Bonus for accurate passes
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "long_pass_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_direction' in o:
                ball_speed = np.linalg.norm(o['ball_direction'][:2])  # Only consider x,y components
                if ball_speed > self.pass_threshold and o['ball_owned_team'] == 1:
                    # Assuming team 1 is the controlled team and the agent decided to do a long pass
                    target_error = np.linalg.norm(o['ball'][:2] - o['right_team'][0][:2])  # Distance to closest opponent
                    if target_error < 0.1:  # Very arbitrary threshold for "accuracy"
                        components['long_pass_reward'][rew_index] = self.accuracy_bonus
                        reward[rew_index] += components['long_pass_reward'][rew_index]

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
