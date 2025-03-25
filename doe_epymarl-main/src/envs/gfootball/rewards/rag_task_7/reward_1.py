import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on the effectiveness of sliding tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.successful_tackles = 0
        self.tackle_attempt_rewards = -0.1
        self.successful_tackle_rewards = 0.5
        self.distance_threshold = 0.1  # distance to consider tackle successful

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.successful_tackles = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['successful_tackles'] = self.successful_tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.successful_tackles = from_pickle['successful_tackles']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_attempt_reward": [0.0] * len(reward),
                      "successful_tackle_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            is_tackle_attempt = o['sticky_actions'][6] == 1  # index for "slide" action

            # Reward for attempting a tackle
            if is_tackle_attempt:
                components["tackle_attempt_reward"][rew_index] = self.tackle_attempt_rewards
                reward[rew_index] += components["tackle_attempt_reward"][rew_index]

            if 'ball_owned_team' in o and o['ball_owned_team'] in [0, 1]:
                # Checking distance to ball when tackle is attempted
                ball_position = np.array(o['ball'][:2])
                player_position = np.array(o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']])
                distance_to_ball = np.linalg.norm(ball_position - player_position)

                # Successful tackle scenario
                if is_tackle_attempt and distance_to_ball < self.distance_threshold:
                    components["successful_tackle_reward"][rew_index] = self.successful_tackle_rewards
                    reward[rew_index] += components["successful_tackle_reward"][rew_index]
                    self.successful_tackles += 1

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
