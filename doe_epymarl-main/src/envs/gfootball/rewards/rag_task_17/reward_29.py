import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for wide midfield responsibilities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_mastered = np.zeros(2, dtype=bool)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.midfield_mastered.fill(False)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'midfield_mastered': self.midfield_mastered}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.midfield_mastered = from_pickle['CheckpointRewardWrapper']['midfield_mastered']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_mastering_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Iterate over each agent's observation
        for i in range(len(observation)):
            o = observation[i]

            # Reward increased sideline activity and transitioning via high passes
            if 'right_team_roles' in o and o['right_team_roles'][o['active']] in [6, 7]:  # 6 and 7 are LM and RM roles
                sideline_position = abs(o['right_team'][o['active']][1])  # y position tells us how far from the center
                high_pass_action = o['sticky_actions'][9]  # index 9 can be a high pass if configured correctly

                # Check if player is near the sidelines and using high pass
                if sideline_position > 0.3 and high_pass_action:
                    components['midfield_mastering_reward'][i] = 1.0
                    self.midfield_mastered[i] = True
                    reward[i] += components['midfield_mastering_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
