import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for dribbling skills improvement, especially facing the goalkeeper."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # Get the latest observations from the environment
        observations = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_bonus": [0.0, 0.0]}
        
        if observations is None:
            return reward, components
        
        for i in range(len(reward)):
            observation = observations[i]
            if observation['ball_owned_team'] == 0 or observation['ball_owned_team'] == 1:
                # Check if the player is close to the goalkeeper and dribbling
                own_goal_pos = 1 if observation['ball_owned_team'] == 1 else -1
                ball_pos = observation['ball'][0]  # x position of the ball
                if own_goal_pos * ball_pos > 0.7:  # ball is close to the opponent's goal
                    player_pos = observation['right_team'][observation['active']] if own_goal_pos == 1 else observation['left_team'][observation['active']]
                    dribbling = observation['sticky_actions'][9]  # is dribbling
                    if dribbling and abs(player_pos[1]) < 0.1:  # near the central y axis
                        components['dribbling_bonus'][i] = 0.2
                        
            reward[i] += components['dribbling_bonus'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
