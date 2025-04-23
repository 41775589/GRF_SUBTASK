import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on midfield dynamics for central and wide roles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Parameters for midfield control rewards
        self.midfield_control_reward = 0.05
        self.pass_accuracy_reward = 0.10
        self.transition_bonus = 0.15

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = 1  # Sparse state representation
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # State restoration not specified as it's a simple count
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_reward": [0.0] * len(reward)}

        # Calculate midfield control rewards based on current observations
        for i in range(len(reward)):
            o = observation[i]

            # Central midfield actions
            if o['left_team_roles'][o['active']] == 5 or o['left_team_roles'][o['active']] == 4:
                if o['ball_owned_player'] == o['active']:
                    components["midfield_reward"][i] += self.midfield_control_reward
            
            # Wide midfield actions
            if o['left_team_roles'][o['active']] == 6 or o['left_team_roles'][o['active']] == 7:
                if o['ball_owned_player'] == o['active']:
                    components["midfield_reward"][i] += self.midfield_control_reward

            # Transition reward for transitioning the ball from defense to forward
            if o['ball_owned_player'] == o['active'] and np.sqrt(o['ball'][0]**2 + o['ball'][1]**2) > 0.3:
                components["midfield_reward"][i] += self.transition_bonus
            
            # Update the final reward
            reward[i] = reward[i] + components["midfield_reward"][i]

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
