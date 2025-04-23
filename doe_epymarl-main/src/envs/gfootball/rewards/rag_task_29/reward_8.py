import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for shot precision in close range of the goal."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
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
                      "precision_shots_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            ball_x, ball_y = o['ball'][:2]
            goal_y_range = (-0.044, 0.044)  # Goal y-coordinate range near the center

            # Calculate the distance from the ball to the center of the opponent's goal.
            distance_to_goal = abs(ball_x - 1.0)  # Assuming the agent is moving towards the right side goal located at x=1.0

            # Check if the ball is near the goal horizontally and within a tight y-range
            if distance_to_goal < 0.1 and goal_y_range[0] < ball_y < goal_y_range[1]:
                if o['ball_owned_team'] == 1 and o['ball_owned_player'] == o['active']:
                    # The coefficient of 0.2 is indicative of the importance of this reward; you might need to tune based on results.
                    shot_reward = 0.2 / (distance_to_goal + 0.01)  # Encourage lower distance with higher reward
                    components["precision_shots_reward"][rew_index] += shot_reward

                reward[rew_index] += components["precision_shots_reward"][rew_index]

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
