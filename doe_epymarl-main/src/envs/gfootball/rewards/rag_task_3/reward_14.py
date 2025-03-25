import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that reinforces learning shooting skills during game simulations."""

    def __init__(self, env):
        super().__init__(env)
        self.shot_attempts = 0
        self.successful_shots = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.shot_attempts = 0
        self.successful_shots = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shot_attempts'] = self.shot_attempts
        to_pickle['successful_shots'] = self.successful_shots
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shot_attempts = from_pickle['shot_attempts']
        self.successful_shots = from_pickle['successful_shots']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shot_accuracy_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        multiplier = 5  # Boost the shot related reward

        for idx in range(len(reward)):
            o = observation[idx]
            if o['game_mode'] == 6:  # 6 corresponds to penalty kicks
                self.shot_attempts += 1
                if o['score'][0] > o['score'][1]:  # Assuming the agent is on the left
                    self.successful_shots += 1
                    components["shot_accuracy_reward"][idx] = multiplier * (self.successful_shots / self.shot_attempts)
                    reward[idx] += components['shot_accuracy_reward'][idx]

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
