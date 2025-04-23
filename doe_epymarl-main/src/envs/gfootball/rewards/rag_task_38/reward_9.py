import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful initiation of counterattacks via long passes."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        self.long_pass_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self.last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle.get('last_ball_position', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0, 0.0]}
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            if self.last_ball_position is not None:
                # Calculate ball displacement
                displacement = np.linalg.norm(np.subtract(self.last_ball_position, o['ball'][:2]))

                # High displacement during a pass suggests a long pass
                if displacement > 0.5:
                    if o['ball_owned_team'] == 1 and self.last_ball_position[0] < 0:
                        # Reward for long pass starting in own half and possession retained
                        components["long_pass_reward"][rew_index] += self.long_pass_reward
                        reward[rew_index] += components["long_pass_reward"][rew_index]

            # Update the recorded last ball position
            self.last_ball_position = o['ball'][:2]
        
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
