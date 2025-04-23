import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense high pass reward from midfield to create direct scoring opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.high_pass_threshold = 0.2  # heuristic threshold for what constitutes a 'high pass'
        self.midfield_x_threshold = 0.2 # values near the midfield region
        self.goal_threshold = 0.8       # proximity to the opponent goal
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
        observation = self.env.unwrapped.observation()
        
        if observation is None:
            return reward, components

        for i in range(len(reward)):
            obs = observation[i]
            has_ball = obs['ball_owned_team'] == 0 and obs['ball_owned_player'] == obs['active']
            ball_height = obs['ball'][2]
            player_x = obs['left_team'][obs['active']][0]
            close_to_goal = player_x >= self.goal_threshold

            # Check if player executed a high pass from midfield
            if has_ball and ball_height > self.high_pass_threshold and abs(player_x) <= self.midfield_x_threshold:
                # Check if the pass leads directly towards the opponent's goal area
                if close_to_goal:
                    components['high_pass_reward'][i] = 0.3
                else:
                    components['high_pass_reward'][i] = 0.1

            reward[i] += components['high_pass_reward'][i]

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
                if action == 1:
                    self.sticky_actions_counter[i] += 1
                    info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
