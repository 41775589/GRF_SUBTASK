import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized checkpoint reward for goalkeeper coordination under high-pressure scenarios."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.goalkeeper_index = 0  # Assuming the goalkeeper is the first player on the team list
        self.sticky_actions_counter = np.zeros(10, dtype=int)
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}

        goalie_reward = np.zeros(len(reward))
        if observation is not None:
            for i, obs in enumerate(observation):
                if obs['game_mode'] in [3, 4] and obs['ball_owned_team'] == 1:  # Checking freekick or corner defense
                    # Reward goalkeeper for being close to the goal and ball not being close to the goal
                    goalie_pos = obs['right_team'][self.goalkeeper_index] if i == 1 else obs['left_team'][self.goalkeeper_index]
                    ball_pos = obs['ball'][:2]
                    goalie_to_ball_distance = np.linalg.norm(goalie_pos - ball_pos)
                    goal_pos = np.array([1, 0]) if i == 1 else np.array([-1, 0])

                    # Goalkeeper is rewarded for minimizing the distance to the goal and maximizing distance from ball during corners or freekicks
                    goalie_goal_distance = np.linalg.norm(goalie_pos - goal_pos)
                    goalie_reward[i] = (1 - goalie_goal_distance) + goalie_to_ball_distance

        new_reward = np.array(reward) + goalie_reward
        components["goalkeeper_pressure_reward"] = list(goalie_reward)
        return new_reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info["components_" + key] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        return observation, reward, done, info
