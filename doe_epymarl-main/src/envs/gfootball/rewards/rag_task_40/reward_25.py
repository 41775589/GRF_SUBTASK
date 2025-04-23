import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense defensive checkpoint reward to enhance defensive capabilities."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self._num_checkpoints = 5  # number of defensive checkpoints
        self._checkpoint_reward = 0.2  # reward for reaching a defensive checkpoint
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_checkpoint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            own_goal = -1 if o["ball_owned_team"] == 0 else 1
            opponent_goal = 1 if own_goal == -1 else -1

            # Check if the current observation concerns defensive action
            if o['ball_owned_team'] == own_goal:
                # Calculate the distance to the defensive goal
                goal_distance = abs(o['ball'][0] - own_goal)

                # Calculate checkpoints based on the distance to own goal
                for i in range(1, self._num_checkpoints + 1):
                    if goal_distance <= abs(own_goal / self._num_checkpoints * i):
                        components["defensive_checkpoint_reward"][rew_index] += self._checkpoint_reward
                        break

                # Augment the reward with the defensive checkpoint rewards
                reward[rew_index] += components["defensive_checkpoint_reward"][rew_index]

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
