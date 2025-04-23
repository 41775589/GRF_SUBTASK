import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """This wrapper adjusts the rewards to promote short passing skill development under defensive pressure.
    It focuses on ball retention and effective distribution strategies to facilitate defensive stability
    and efficient transitions to counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owned_player = None
        self.pass_completion_counter = 0
        self.pass_reward = 0.5
        self.ball_ownership_reward = 0.1
        self.under_pressure_reward = 0.3

    def reset(self):
        super().reset()
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owned_player = None
        self.pass_completion_counter = 0
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "ball_ownership_reward": [0.0] * len(reward),
                      "under_pressure_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Encourage retaining possession
            if o['ball_owned_team'] == 0:  # Assuming team 0 is the agent's team
                if self.previous_ball_owned_player is not None and \
                   self.previous_ball_owned_player != o['ball_owned_player']:
                    self.pass_completion_counter += 1
                    components["pass_reward"][rew_index] = self.pass_reward
                components["ball_ownership_reward"][rew_index] = self.ball_ownership_reward

                # Additional reward for passing under pressure
                if o['game_mode'] in {2, 3, 4, 5, 6}:  # game modes that imply opponent interaction
                    components["under_pressure_reward"][rew_index] = self.under_pressure_reward

                reward[rew_index] += sum(components[c][rew_index] for c in components)

            self.previous_ball_owned_player = o['ball_owned_player']

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
