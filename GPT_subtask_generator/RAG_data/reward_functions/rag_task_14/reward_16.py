import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for a sweeper task."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_reward = 1.0  # Reward for clearing the ball from close to own goal
        self.tackle_reward = 0.5    # Reward for tackling while last man
        self.support_reward = 0.3   # Reward for supporting actions

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "sweeper_rewards": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                # Considering the player is a sweeper at this moment.
                # Clearing action near our goal area.
                if np.linalg.norm(np.array(o['ball']) - np.array([-1, 0])) < 0.3:
                    components["sweeper_rewards"][i] += self.clearance_reward

                # Successful tackle as last man (distance measurement is an example and basic).
                if o['left_team_active'][o['active']] and np.any(o['right_team'][:, 0] > o['left_team'][o['active'], 0]):
                    components["sweeper_rewards"][i] += self.tackle_reward

                # Support action by maintaining position behind the ball carrier or blocking pass lanes.
                if o['ball'][0] < 0:  # Assuming ball is in our half
                    if abs(o['ball'][1] - o['left_team'][o['active'], 1]) < 0.1:
                        components["sweeper_rewards"][i] += self.support_reward

                # Incorporate the rewards and penalties.
                reward[i] += components["sweeper_rewards"][i]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
