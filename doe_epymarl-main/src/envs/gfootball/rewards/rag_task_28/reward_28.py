import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a specialized reward based on dribbling and feint tactics,
    particularly when facing the goalkeeper in face-to-face situations.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Parameters for dribble rewards
        self.dribble_reward = 0.1
        self.feint_reward = 0.2
        self.pressure_factor = 0.05
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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

        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "special_dribble_reward": [0.0] * len(reward),
            "feint_and_pressure_reward": [0.0] * len(reward)
        }

        # Loop through each agent's observation
        for agent_index in range(len(reward)):
            agent_obs = observation[agent_index]

            if agent_obs['ball_owned_team'] == 0:  # If the left team (controlled by our agent) owns the ball
                # Increase reward for dribbling (action index 9)
                if agent_obs['sticky_actions'][9] == 1:  # Dribbling action
                    components["special_dribble_reward"][agent_index] += self.dribble_reward

                # Simulate conditions for feint and direction changes under pressure (near opponent's goal)
                opponent_goal = 1  # Assuming right team's goal as x = 1
                distance_to_goal = abs(agent_obs['ball'][0] - opponent_goal)

                if distance_to_goal < 0.2:  # Close to goal, hence under pressure
                    direction_changes = np.sum(np.abs(np.diff(agent_obs['left_team_direction'], axis=0)))
                    components["feint_and_pressure_reward"][agent_index] += self.feint_reward * direction_changes
        
        # Compute final reward
        for idx in range(len(reward)):
            reward[idx] += components["special_dribble_reward"][idx] + self.pressure_factor * components["feint_and_pressure_reward"][idx]

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return obs, reward, done, info
