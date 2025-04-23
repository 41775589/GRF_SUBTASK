import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for initiating counterattacks through accurate long passes and quick transitions from defense to attack.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_initial_position = None
        self.counter_attack_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_initial_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['ball_initial_position'] = self.ball_initial_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_initial_position = from_pickle['ball_initial_position']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counter_attack_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball'][0]  # Reading the x-coordinate of the ball.

            # Record the initial position of the ball when the team regains control
            if o['ball_owned_team'] == 0 and self.ball_initial_position is None:
                self.ball_initial_position = ball_position
            
            # Check if a long pass has been made by detecting a significant movement of ball in x direction.
            if self.ball_initial_position is not None and abs(ball_position - self.ball_initial_position) > 0.3:
                # Only reward if the ball is still owned by the team to ensure it was a successful pass.
                if o['ball_owned_team'] == 0:
                    components["counter_attack_reward"][rew_index] = self.counter_attack_reward
                    reward[rew_index] += self.counter_attack_reward
                # Reset the initial ball position to capture subsequent counterattacks.
                self.ball_initial_position = None

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
