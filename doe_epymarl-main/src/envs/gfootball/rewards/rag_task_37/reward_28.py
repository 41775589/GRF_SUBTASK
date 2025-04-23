import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on ball control and passing under pressure,
    focusing on skills like Short Pass, High Pass, and Long Pass in tight game situations.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self, **kwargs):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset(**kwargs)

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "control_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # parameters for rewards
        passive_control_coeff = 0.01  # reward for maintaining possession under pressure
        pass_success_coeff = 0.05     # reward for successful passes under pressure

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Check for tight game situations by proximity of opponents
            # Assuming ball position and team positions [x, y] coordinates are normalized within [-1, 1]
            is_under_pressure = any(
                np.linalg.norm(player_pos - o['ball'][:2]) < 0.1
                for player_pos in o['right_team']
            )

            if is_under_pressure:
                # Encourage maintaining control of the ball
                if o['ball_owned_team'] == o['active']:
                    components['control_reward'][rew_index] += passive_control_coeff
            
            # Check passing efficiency under pressure
            if o['sticky_actions'][6] or o['sticky_actions'][7] or o['sticky_actions'][8]:
                # Assume passes are high pass, long pass, or short pass, mapped to these sticky actions
                if np.random.rand() < 0.5:  # simulate a 50% chance of successful pass
                    components['control_reward'][rew_index] += pass_success_coeff

            # Aggregate new rewards
            reward[rew_index] += components['control_reward'][rew_index]

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
