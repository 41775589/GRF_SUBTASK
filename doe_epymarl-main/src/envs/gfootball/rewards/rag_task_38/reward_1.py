import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds rewards for initiating counterattacks post-defense through long passes and quick transitions.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_reward = 1.0  # Reward for successful long pass
        self.quick_transition_reward = 1.5  # Reward for quick transition

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
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward),
                      "quick_transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_reward = reward[rew_index]
            ball_direction = o['ball_direction']
            left_team = o['left_team']
            left_team_direction = o['left_team_direction']
            ball_owned_team = o['ball_owned_team']
            
            # Reward for long passes from defense to attack transition
            if ball_owned_team == 0:  # If left_team has the ball
                own_goal = -1
                opponent_goal = 1
                ball_pos_x = o['ball'][0]
                if ball_pos_x < own_goal and any(ball_direction):
                    # Check if there's a significant forward ball movement
                    if ball_direction[0] > 0.1:
                        components['long_pass_reward'][rew_index] = self.long_pass_reward
                        reward[rew_index] += self.long_pass_reward
            
            # Reward for quick transitions
            # Assuming quick transitions need instant direction change or fast movement to the opponent half
            if ball_owned_team == 0:  # If still left_team has the ball
                for i, player_pos in enumerate(left_team):
                    if -0.5 < player_pos[0] < 0.0:  # Players in own half
                        if left_team_direction[i][0] > 0.1:  # Moving forward fast
                            components['quick_transition_reward'][rew_index] = self.quick_transition_reward
                            reward[rew_index] += self.quick_transition_reward

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
