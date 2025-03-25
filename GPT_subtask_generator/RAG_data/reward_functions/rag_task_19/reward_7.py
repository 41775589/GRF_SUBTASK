import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for maintaining defensive position and controlling midfield effectively."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.defensive_positions = {}
        self.midfield_control = {}
        self.defensive_reward = 0.2
        self.control_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.defensive_positions = {}
        self.midfield_control = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_positions'] = self.defensive_positions
        to_pickle['midfield_control'] = self.midfield_control
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positions = from_pickle.get('defensive_positions', {})
        self.midfield_control = from_pickle.get('midfield_control', {})
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward

        base_reward = reward.copy()
        reward_components = {"base_score_reward": base_reward,
                             "defensive_position_reward": [0] * len(reward),
                             "midfield_control_reward": [0] * len(reward)}

        for index, obs in enumerate(observation):
            # Defensive effectiveness: consider position of defensive players
            if obs['left_team_roles'][obs['active']] in [1, 2, 3, 4]:  # CB, LB, RB, DM roles
                distance_from_goal = abs(obs['left_team'][obs['active']][0] + 1)
                if distance_from_goal < 0.3:  # Close to own goal area
                    if not self.defensive_positions.get(index):
                        self.defensive_positions[index] = True
                        reward[index] += self.defensive_reward
                        reward_components["defensive_position_reward"][index] = self.defensive_reward

            # Midfield control: consider ball position and possession in midfield zone
            if -0.3 <= obs['ball'][0] <= 0.3 and obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']:
                if obs['left_team_roles'][obs['active']] in [5, 6, 7, 8]:  # CM, LM, RM, AM roles
                    if not self.midfield_control.get(index):
                        self.midfield_control[index] = True
                        reward[index] += self.control_reward
                        reward_components["midfield_control_reward"][index] = self.control_reward

        return reward, reward_components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        final_reward = sum(reward)
        info['final_reward'] = final_reward
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
