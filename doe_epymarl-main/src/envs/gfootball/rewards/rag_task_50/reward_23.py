import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for executing long and accurate passes across predefined checkpoints."""

    def __init__(self, env, num_checkpoints=5):
        super().__init__(env)
        self.num_checkpoints = num_checkpoints
        self.long_pass_reward = 0.1
        self.checkpoints_collected = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.checkpoints_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.checkpoints_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.checkpoints_collected = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0]*len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            # Calculate the distance of the most recent pass, assuming ball movement represents passes
            if 'ball_direction' in obs:
                distance = np.linalg.norm(obs['ball_direction'])
                previous_distance = self.checkpoints_collected.get(idx, 0)

                # Define checkpoints by specific long pass distances
                for checkpoint in range(1, self.num_checkpoints + 1):
                    threshold = checkpoint * (1 / self.num_checkpoints) * 2  # Normalized for field scale [-1, 1]
                    if previous_distance < threshold <= distance:
                        components["long_pass_reward"][idx] += self.long_pass_reward
                        reward[idx] += components["long_pass_reward"][idx]
                self.checkpoints_collected[idx] = distance

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions counter in the environment information.
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = max(self.sticky_actions_counter[i], action_active)

        return observation, reward, done, info
