import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards technical skills for executing high passes with precision in football."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define some arbitrary height as 'high' pass (for example, z>0.2 in ball coordinate)
        self.high_pass_height_threshold = 0.2
        self.high_pass_skill_reward = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_skill_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            height_of_ball_pass = o['ball'][2]
            game_mode = o['game_mode']

            # Reward high passes in relevant game modes
            if height_of_ball_pass > self.high_pass_height_threshold:
                # Focus on game modes where high passes are typical and useful
                if game_mode in [0, 1, 2, 3, 5]:  # Normal play, kick off, goal kick, free kick, throw in
                    components["high_pass_skill_reward"][rew_index] = self.high_pass_skill_reward
                    reward[rew_index] += components["high_pass_skill_reward"][rew_index]

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
