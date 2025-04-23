import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards defensive actions, specifically focusing on 
    interception and defensive positioning under high-pressure scenarios.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.interception_reward = 0.2
        self.defensive_positioning_reward = 0.1
        self.high_pressure_defense_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment state and the count of sticky actions.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Get the state of the wrapper including base environment states.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state of the environment from the given state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        """
        Calculate additional reward based on defensive actions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "interception_reward": [0.0] * len(reward), 
                      "defensive_positioning_reward": [0.0] * len(reward),
                      "high_pressure_defense_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward interception
            if o['game_mode'] in {2, 3, 4, 5, 6}: # Modes where the ball is more likely to change possession
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                    components["interception_reward"][rew_index] = self.interception_reward

            # Reward defensive positioning, particularly when close to own goal under threat
            own_goal_x = -1 if o['ball_owned_team'] == 1 else 1
            distance_to_goal = np.linalg.norm(o['left_team'][o['active']] - np.array([own_goal_x, 0]))
            if distance_to_goal < 0.2:
                components["defensive_positioning_reward"][rew_index] = self.defensive_positioning_reward

            # Additional reward for successful defense under high pressure (ball is close and in play)
            if o['ball'][2] < 0.1: # Assuming z < 0.1 means close to the ground and in play
                components["high_pressure_defense_reward"][rew_index] = self.high_pressure_defense_reward

        # Compute the total reward with components
        for rew_index in range(len(reward)):
            reward[rew_index] += components["interception_reward"][rew_index] + components["defensive_positioning_reward"][rew_index] + components["high_pressure_defense_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Step through the environment by applying the action and calculating rewards.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Counter for sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
