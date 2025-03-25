import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focusing on defensive actions like intense man-marking, blocking shots,
    and interrupting the forward moves of the opposing players."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.block_reward = 0.1
        self.intercept_reward = 0.05
        self.previous_ball_position = None
        self.previous_player_position = None
        self.previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_position = None
        self.previous_player_position = None
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['previous_ball_position'] = self.previous_ball_position
        to_pickle['previous_player_position'] = self.previous_player_position
        to_pickle['previous_ball_owner'] = self.previous_ball_owner
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_position = from_pickle['previous_ball_position']
        self.previous_player_position = from_pickle['previous_player_position']
        self.previous_ball_owner = from_pickle['previous_ball_owner']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "block_reward": [0.0] * len(reward),
                      "intercept_reward": [0.0] * len(reward)}

        for i in range(len(reward)):
            current_ball_position = observation[i]['ball']
            current_player_position = observation[i]['right_team'] if observation[i]['ball_owned_team'] == 1 else observation[i]['left_team']
            ball_owner = observation[i]['ball_owned_team']

            # Reward for blocking shots: if the player is between the ball and the goal and ball is shot by opponent
            if ball_owner != 0 and self.previous_ball_owner != ball_owner:
                ball_direction = observation[i]['ball_direction'][:2]
                goal_position = np.array([1, 0]) if ball_owner == 1 else np.array([-1, 0])
                ball_to_goal = goal_position - current_ball_position[:2]
                if np.dot(ball_direction, ball_to_goal) > 0:  # ball is moving towards the goal
                    for player_pos in current_player_position:
                        if np.linalg.norm(player_pos - current_ball_position[:2]) < 0.1:  # player is close to ball
                            components["block_reward"][i] = self.block_reward
                            reward[i] += components["block_reward"][i]

            # Reward for intercepting passes
            if self.previous_ball_position is not None and self.previous_ball_owner != 0:
                travel_distance = np.linalg.norm(current_ball_position[:2] - self.previous_ball_position)
                if travel_distance > 0.2:  # assuming a pass
                    player_distances = np.linalg.norm(current_player_position - self.previous_ball_position, axis=1)
                    if np.min(player_distances) < 0.1:  # player intercepts the pass
                        components["intercept_reward"][i] = self.intercept_reward
                        reward[i] += components["intercept_reward"][i]

            self.previous_ball_position = current_ball_position
            self.previous_player_position = current_player_position
            self.previous_ball_owner = ball_owner

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
