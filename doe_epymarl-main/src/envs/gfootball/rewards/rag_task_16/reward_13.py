import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper focusing on enhancing the ability to perform high passes accurately and effectively.
    This rewards agents for executing high passes that reach a teammate, over a certain height.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define height threshold for considering a pass as "high"
        self.high_pass_height_threshold = 0.1  # Arbitrary height threshold
        # Reward coefficients
        self.high_pass_reward = 0.5
        self.catch_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper_sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('CheckpointRewardWrapper_sticky_actions', np.zeros(10))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "high_pass_reward": [0.0] * len(reward),
                      "catch_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Check if ball is owned by the agent's team
            if o['ball_owned_team'] != -1 and o['ball'][2] > self.high_pass_height_threshold:
                if self.env.unwrapped.previous_action[rew_index] in [9]:  # Action index 9 is assumed to be 'high pass'
                    components["high_pass_reward"][rew_index] = self.high_pass_reward
                    
            # Check if the ball has successfully been passed to a teammate
            if o['ball_owned_team'] == 0: # Assuming 0 is the team ID for the agent's team
                # Ensure the ball was previously high enough and now controlled by a teammate
                if o['ball'][2] < self.high_pass_height_threshold:
                    components["catch_reward"][rew_index] = self.catch_reward

            # Calculate total reward for this agent
            reward[rew_index] = reward[rew_index] \
                                + components["high_pass_reward"][rew_index] \
                                + components["catch_reward"][rew_index]

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
