import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds long-range scoring training. It rewards attempting shots from outside the penalty box,
    particularly for long-distance shots aimed at overcoming the defenders.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.distance_threshold = 0.6  # Roughly outside the penalty box in normalized field coordinates

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        
        components = {"base_score_reward": reward.copy(),
                      "long_shot_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            player_info = observation[rew_index]
            distance_to_goal = np.linalg.norm(
                player_info['ball'][:2] - np.array([1, 0]) if player_info['ball_owned_team'] == 0 else np.array([-1, 0])
            )
            shot_attempted = player_info['sticky_actions'][9]  # Assuming index 9 is shooting action

            # Giving reward for shots taken from long distance outside the penalty box
            if distance_to_goal > self.distance_threshold and shot_attempted:
                components["long_shot_reward"][rew_index] = 0.1  # Assign reward for attempting long-range shots
                reward[rew_index] += components["long_shot_reward"][rew_index]
        
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
