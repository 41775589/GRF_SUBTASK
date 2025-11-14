import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for learning coordinated offensive strategies."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        stored = from_pickle.get('CheckpointRewardWrapper', {})
        self.sticky_actions_counter = stored.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_coordination_reward": np.zeros_like(reward)}

        if observation is None:
            return reward, components

        # Enhancing offensive strategies
        for rew_index, o in enumerate(observation):
            # Reward for maintaining possession of the ball
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:  # Assuming controlled team is '1'
                components["offensive_coordination_reward"][rew_index] += 0.1

            # Reward for player getting closer to the opposing goal
            goal_distance = o['ball'][0]   # Assuming the goal is at the 1.0 x-coordinate
            components["offensive_coordination_reward"][rew_index] += (1 - goal_distance) * 0.05

            # Reward for successful passes within the opposing half
            if 'game_mode' in o and o['game_mode'] in {1, 3}:  # Game modes where a pass might have completed
                if o['ball'][0] > 0.5:  # Ball in the opponent's half
                    components["offensive_coordination_reward"][rew_index] += 0.2

            reward[rew_index] += components["offensive_coordination_reward"][rew_index]

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
