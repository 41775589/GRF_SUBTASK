import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for counterattacks through accurate long passes and quick transitions from defense to attack."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.counterattack_reward = 0.2
        self.long_pass_reward = 0.1
        self.quick_transition_reward = 0.1
        self.long_pass_threshold = 0.2  # Threshold for considering a pass as 'long'

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)
    
    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counterattack_reward": [0.0] * len(reward),
                      "long_pass_reward": [0.0] * len(reward),
                      "quick_transition_reward": [0.0] * len(reward)}

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if o['game_mode'] in [3, 4, 5, 6]:  # Transition moments (e.g., free kicks, corners, etc.)
                if o['ball_owned_team'] == 0:
                    components['counterattack_reward'][rew_index] = self.counterattack_reward
                    reward[rew_index] += self.counterattack_reward

            previous_ball_position = o['ball'] - o['ball_direction']
            ball_travel_distance = np.linalg.norm(o['ball_direction'][:2])
            if ball_travel_distance > self.long_pass_threshold:
                components['long_pass_reward'][rew_index] = self.long_pass_reward
                reward[rew_index] += self.long_pass_reward

            if 'active' in o and o['sticky_actions'][9]:  # Dribbling involved in transition
                components['quick_transition_reward'][rew_index] = self.quick_transition_reward
                reward[rew_index] += self.quick_transition_reward

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
        return observation, reward, done, info
