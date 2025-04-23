import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for defensive actions and strategic 
    positioning to handle direct attacks effectively and setup counterattacks.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define your customized reward components here.
        # Setup rewards for interception, position holding near one's goal
        self.interception_reward = 0.2
        self.good_positioning_reward = 0.1
        self.threshold_distance = 0.5  # Ideal distance threshold

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "interception_reward": [0.0] * len(reward),
                      "good_positioning_reward": [0.0] * len(reward)}

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for intercepting the ball from the opposing team
            if o['game_mode'] == 0 and o['ball_owned_team'] == 1 - o['left_team']:  # Assuming 'left_team' is the side the agent defends
                components['interception_reward'][rew_index] = self.interception_reward
                reward[rew_index] += components['interception_reward'][rew_index]

            # Additional reward for being well-positioned defensively near the own goal
            if o['left_team']:
                x_pos = -1  # Position of the left team's goal
            else:
                x_pos = 1   # Position of the right team's goal

            player_pos = o['left_team'][o['active']]
            distance_to_goal = abs(player_pos[0] - x_pos)
            if distance_to_goal <= self.threshold_distance:
                components["good_positioning_reward"][rew_index] = self.good_positioning_reward
                reward[rew_index] += components["good_positioning_reward"][rew_index]

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
