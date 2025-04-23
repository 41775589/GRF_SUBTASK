import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards for midfield dynamics mastery with different player roles in football."""

    def __init__(self, env):
        super().__init__(env)
        # Reward coefficients
        self._midfield_control_reward = 0.05
        self._forward_support_reward = 0.05
        self._defensive_action_reward = 0.03
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "midfield_control_reward": [0.0] * len(reward),
                      "forward_support_reward": [0.0] * len(reward),
                      "defensive_action_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # iterate through all agents' observations
        for i, obs in enumerate(observation):
            # Assign roles based on left team roles mapped as [GK, CBs, LB, RB, DMs, CMs, LMs, RMs, AMs, CF]
            midfielder_roles = {4, 5, 6, 7, 8}  # Assuming DM, CM, LM, RM, AM as midfield roles
            if obs['active'] in midfielder_roles:
                # Control midfield: reward for maintaining position in midfield areas and having the ball
                if obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                    components["midfield_control_reward"][i] = self._midfield_control_reward
                
                # Forward support: reward for passing the ball towards forward players or moving towards the opponent's half
                if obs['ball_direction'][0] > 0:  # Assuming positive x-direction is towards opponents goal
                    components["forward_support_reward"][i] = self._forward_support_reward

                # Defensive backing: reward for taking positions between the ball and own goal when the other team has the ball
                if obs['ball_owned_team'] == 1:  # Ball owned by the opponents
                    own_goal_x = -1  # Assuming the own goal is at x = -1
                    if abs(obs['ball'][0] - own_goal_x) > abs(obs['left_team'][obs['active']][0] - own_goal_x):
                        components["defensive_action_reward"][i] = self._defensive_action_reward

            # Sum up the additional rewards
            reward[i] += sum([components[key][i] for key in components.keys() if key != "base_score_reward"])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
