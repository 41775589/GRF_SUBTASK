import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for offensive strategies in football playing."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define bonus reward coefficients
        self.shooting_coefficient = 0.3
        self.dribbling_coefficient = 0.2
        self.passing_coefficient = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward),
                      "dribbling_reward": [0.0] * len(reward),
                      "passing_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Check if scoring a goal has just occurred
            if reward[rew_index] == 1 or o['ball'][0] > 0.9:  # close to opponent's goal
                components['shooting_reward'][rew_index] = self.shooting_coefficient

            # Check if significant dribbling action is happening
            if o['sticky_actions'][9]:  # dribble action active
                components['dribbling_reward'][rew_index] = self.dribbling_coefficient

            # Check for passing - kick without change of possession
            if o['ball_owned_team'] == 0 and np.linalg.norm(o['ball_direction'][:2]) > 0.1:
                if 'action' in o and (o['action_to_release'] == 6 or o['action_to_release'] == 7):  # long pass or high pass
                    components['passing_reward'][rew_index] = self.passing_coefficient

            # Sum up rewards for the current step
            total_reward = (reward[rew_index] +
                            components['shooting_reward'][rew_index] +
                            components['dribbling_reward'][rew_index] +
                            components['passing_reward'][rew_index])

            reward[rew_index] = total_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    if action:
                        self.sticky_actions_counter[i] += 1
                        info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
