import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to augment the reward based on defensive actions and transitions to counter-attacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.interception_bonus = 0.1  # bonus for intercepting the ball
        self.counter_attack_bonus = 0.2  # bonus for valid counter-attack moves
        self.last_ball_owner = None  # track last ball ownership for transition checks
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.last_ball_owner = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'last_ball_owner': self.last_ball_owner
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_owner = from_pickle['CheckpointRewardWrapper']['last_ball_owner']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "interception_bonus": [0.0] * len(reward),
                      "counter_attack_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i, obs in enumerate(observation):
            # Check for interception
            if obs['ball_owned_team'] == 1 and self.last_ball_owner == 0:
                components["interception_bonus"][i] = self.interception_bonus
                reward[i] += components["interception_bonus"][i]

            # Check for counter-attacks
            if obs['ball_owned_team'] == 1 and 'active' in obs and obs['active']:
                if components["interception_bonus"][i]:
                    # Award counter-attack bonus if following an interception
                    components["counter_attack_bonus"][i] = self.counter_attack_bonus
                    reward[i] += components["counter_attack_bonus"][i]

            # Update last ball owner status
            self.last_ball_owner = obs['ball_owned_team']

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
