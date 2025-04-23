import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for offensive skills like passing, shooting, 
    dribbling, and sprinting which are essential for creating scoring opportunities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        
        # Define the reward coefficients for various offensive actions.
        self.pass_reward = 0.05
        self.shot_reward = 0.1
        self.dribble_reward = 0.03
        self.sprint_reward = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward),
                      "shot_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "sprint_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check for sticky_actions and reward accordingly.
            sticky_actions = o['sticky_actions']
            
            if sticky_actions[6] == 1 or sticky_actions[7] == 1:  # Short pass or Long pass
                components["pass_reward"][rew_index] = self.pass_reward
            if sticky_actions[2] == 1:  # Shot
                components["shot_reward"][rew_index] = self.shot_reward
            if sticky_actions[9] == 1:  # Dribble
                components["dribble_reward"][rew_index] = self.dribble_reward
            if sticky_actions[8] == 1:  # Sprint
                components["sprint_reward"][rew_index] = self.sprint_reward
            
            reward[rew_index] += (components["pass_reward"][rew_index] +
                                  components["shot_reward"][rew_index] +
                                  components["dribble_reward"][rew_index] +
                                  components["sprint_reward"][rew_index])

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
