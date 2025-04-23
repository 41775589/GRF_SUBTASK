import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focused on encouraging playing behavior typical for wide midfielders, emphasizing high passing,
    positioning to support lateral transitions, and creating space to stretch the opponent's defense."""
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track sticky actions
        self.high_pass_reward = 0.2
        self.positioning_reward = 0.1

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
                      "high_pass_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Adding rewards for high pass action; identify high pass from the actions (sticky_actions[2] might represent high pass in some configurations)
            if o['sticky_actions'][2] == 1:
                components["high_pass_reward"][rew_index] = self.high_pass_reward

            # Positional reward for wide midfielders, encourage positions close to sidelines to stretch play
            x, y = o['left_team'][o['active']]
            if abs(y) > 0.3:  # the player is wide enough to stretch the defense
                components["positioning_reward"][rew_index] = self.positioning_reward

            # Aggregate the rewards
            reward[rew_index] += (components["high_pass_reward"][rew_index] +
                                  components["positioning_reward"][rew_index])

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
