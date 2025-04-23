import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds rewards based on mid to long-range precise ball passes."""
    
    def __init__(self, env):
        super().__init__(env)
        self.passing_threshold = 0.3  # Defines what is considered a long pass
        self.pass_completion_reward = 0.2
        self.high_pass_multiplier = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}  # Capture any specific state here if needed
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # Recover state if anything was specifically stored in get_state
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'pass_completion_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_position = o['ball']
            ball_direction = o['ball_direction']
            last_ball_position = ball_position - ball_direction
            ball_distance_moved = np.linalg.norm(ball_direction[:2])

            # Check for long passes
            if ball_distance_moved > self.passing_threshold:
                if o['ball_owned_team'] == 1:  # Assuming the team 1 is our team
                    ball_owner = o['ball_owned_player']
                    if ball_owner is not None:
                        components['pass_completion_reward'][rew_index] = self.pass_completion_reward
                        reward[rew_index] += components['pass_completion_reward'][rew_index] * (1 + self.high_pass_multiplier * (ball_direction[2] > 0.1))
        
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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
