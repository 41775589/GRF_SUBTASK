import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that emphasizes ball control and efficient passing in tight situations.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.pass_quality_reward = 0.5  # Reward for successful pass under pressure
        self.control_under_pressure_reward = 0.3  # Reward for maintaining ball under pressure
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_quality_reward": [0.0] * len(reward),
                      "control_under_pressure_reward": [0.0]* len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check situation under pressure
            if o['game_mode'] in {0, 5} and o['ball_owned_team'] == 0:  # Normal play or throw-in
                if o['ball_owned_player'] == o['active']:
                    num_close_opponents = sum(np.sqrt(np.sum((np.array(o['right_team']) - o['ball'][:2])**2, axis=1)) < 0.1)
                    if num_close_opponents > 2:  # Tight situation: 3 or more opponents nearby
                        components['control_under_pressure_reward'][rew_index] = self.control_under_pressure_reward
                        reward[rew_index] += 1.2 * components['control_under_pressure_reward'][rew_index]

            # Rewarding passing under pressure
            if o['sticky_actions'][6] or o['sticky_actions'][7]:  # high pass or long pass actions
                num_close_opponents = sum(np.sqrt(np.sum((np.array(o['right_team']) - o['ball'][:2])**2, axis=1)) < 0.2)
                if num_close_opponents > 1:  # Assuming the presence of pressure
                    components['pass_quality_reward'][rew_index] = self.pass_quality_reward
                    reward[rew_index] += components['pass_quality_reward'][rew_index]

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
