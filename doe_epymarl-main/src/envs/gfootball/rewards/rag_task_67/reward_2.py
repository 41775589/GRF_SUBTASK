import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for transition skills from defense to attack."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_rewards = 0.05
        self.dribble_rewards = 0.03
        self.max_ball_control_steps = 100  # Arbitrary limit to count control time
        self.ball_control_count = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_control_count = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        # Add checkpoint state 
        to_pickle['ball_control_count'] = self.ball_control_count
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_control_count = from_pickle['ball_control_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "transition_reward": 0.0,
                      "end_ball_control_penalty": 0.0}
        if observation is None:
            return reward, components

        o = observation[0]  # Assuming a single agent
        if o['ball_owned_team'] == 0:  # Team 0 is the controlled team
            self.ball_control_count += 1
            controlled_player = o['designated']
            actions = o['sticky_actions']

            if actions[8]:  # action_dribble is active
                reward += self.dribble_rewards
                components["transition_reward"] += self.dribble_rewards

            if actions[9] and o['game_mode'] == 0:  # action_pass completed in normal game mode
                reward += self.pass_rewards
                components["transition_reward"] += self.pass_rewards

        if self.ball_control_count >= self.max_ball_control_steps:
            # Penalize for long possession without effective transitioning
            end_penalty = -0.1
            reward += end_penalty
            components["end_ball_control_penalty"] = end_penalty
            self.ball_control_count = 0  # Reset counter after penalization

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = reward
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
