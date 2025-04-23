import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards focused on offensive strategies between midfielders
    for space creation, ball delivery, and strikers for finishing plays.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        # Initialize sticky actions counter to observe player behavior continuity
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the sticky actions counter upon environment reset
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Save the current state of the wrapper along with the environment state
        """
        to_pickle['CheckpointRewardWrapper'] = {"sticky_actions_counter": self.sticky_actions_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Restore the state of the wrapper from saved state
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['CheckpointRewardWrapper']['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        """
        Modify the rewards based on offensive strategies between midfielders and strikers.
        """
        # Retrieve the observation from the environment
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "offensive_play_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # Reward midfielders moving towards creating spaces and delivering the ball
            if 'left_team_roles' in o and o['left_team_roles'][o['active']] in [4, 5, 6, 7, 8]:
                dist_to_goal = abs(o['ball'][0] - 1)  # X-coordinate distance to opponent's goal
                if o['ball_owned_team'] == 0:
                    components["offensive_play_reward"][rew_index] = (0.1 / dist_to_goal) if dist_to_goal != 0 else 0.1
                    reward[rew_index] += 1.5 * components["offensive_play_reward"][rew_index]

            # Reward strikers for finishing
            if 'left_team_roles' in o and o['left_team_roles'][o['active']] in [9]:
                if o['ball_owned_team'] == 0:
                    prox_goal_x = 1 - abs(o['ball'][0])
                    prox_goal_y = abs(o['ball'][1])
                    goal_proximity = math.sqrt(prox_goal_x**2 + prox_goal_y**2)
                    if goal_proximity < 0.1:  # significant proximity to goal
                        components["offensive_play_reward"][rew_index] = 0.3
                        reward[rew_index] += 2 * components["offensive_play_reward"][rew_index]

        return reward, components

    def step(self, action):
        """
        Take an action using the wrapped environment and modify the reward returned.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
