import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for both defensive and offensive strategic positioning and transitioning.
    It also rewards appropriate agent movement based on game states like ball possession and game mode.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_positions_reward = 0.05
        self.offensive_positions_reward = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "offensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball']
            player_position = o['left_team'][o['active']] if o['ball_owned_team'] == 0 else o['right_team'][o['active']]

            # add defensive rewards if the team does not have the ball
            if o['ball_owned_team'] != o['right_team']:
                if np.linalg.norm(ball_position - player_position) < 0.2:
                    components["defensive_reward"][rew_index] = self.defensive_positions_reward

            # add offensive rewards if the team has the ball
            if o['ball_owned_team'] == o['right_team']:
                # Encourage attacking towards the opponent's goal
                distance_to_goal = np.abs(player_position[0] - 1)
                if distance_to_goal < 0.2:
                    components["offensive_reward"][rew_index] = self.offensive_positions_reward

            # apply the additional rewards
            reward[rew_index] += (components["defensive_reward"][rew_index] +
                                  components["offensive_reward"][rew_index])

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
