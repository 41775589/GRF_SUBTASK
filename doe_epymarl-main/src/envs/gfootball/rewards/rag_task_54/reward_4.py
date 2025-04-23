import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for effective passing leading to goal opportunities,
    focusing on collaboration between passers (midfielders) and shooters (forward players).
    """

    def __init__(self, env):
        super().__init__(env)
        self.shooter_reward = 0.5
        self.passer_reward = 0.3
        self.goal_reward = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
                      "passer_reward": [0.0] * len(reward),
                      "shooter_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        # Encourage pass efficiency and shooting leading to scoring
        for rew_index, rew in enumerate(reward):
            current_obs = observation[rew_index]
            active_player = current_obs['active']

            # Evaluate if the action was a successful pass leading to scoring opportunity
            if current_obs['ball_owned_player'] == active_player:
                if current_obs['right_team_roles'][active_player] in [4, 5, 6, 7, 8]:  # midfielders
                    components["passer_reward"][rew_index] = self.passer_reward

                if current_obs['left_team_roles'][active_player] in [9] or current_obs['right_team_roles'][active_player] in [9]:  # forwards/strikers
                    # If close to the goal
                    distance_to_goal = abs(current_obs['ball'][0] - 1)  # distance to right goal
                    if distance_to_goal < 0.1:
                        components["shooter_reward"][rew_index] = self.shooter_reward

            # Adjust rewards
            reward[rew_index] += components["passer_reward"][rew_index]
            reward[rew_index] += components["shooter_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
