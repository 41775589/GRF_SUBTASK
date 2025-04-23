import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a task-specific reward for applying Stop-Dribble under pressure.
    The reward encourages stopping and controlling the ball accurately as a defensive tactic.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        new_rewards = np.array(reward, copy=True)

        components = {
            "base_score_reward": reward.copy(),
            "stop_dribble_control_reward": [0.0] * len(reward)
        }

        if observation is None:
            return new_rewards, components

        for i in range(len(reward)):
            obs = observation[i]
            if obs['sticky_actions'][9] == 1 and obs['ball_owned_team'] == 1:
                # Checking if dribble action is active and the ball is owned by right team
                if obs['right_team_active'][obs['active']]:
                    player_coords = obs['right_team'][obs['active']]
                    ball_coords = obs['ball'][:2]

                    distance = np.sqrt((player_coords[0] - ball_coords[0]) ** 2 + (player_coords[1] - ball_coords[1]) ** 2)
                    # Encourage player to stop dribbling when very close to the ball.
                    if distance < 0.1:
                        components['stop_dribble_control_reward'][i] = 0.5
            new_rewards[i] += components['stop_dribble_control_reward'][i]

        return new_rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        new_rewards, components = self.reward(reward)
        info["final_reward"] = sum(new_rewards)

        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, new_rewards, done, info
