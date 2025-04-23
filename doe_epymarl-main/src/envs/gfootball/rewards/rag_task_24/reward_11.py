import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes effective mid to long-range passing as part of coordinated play.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.long_pass_threshold = 0.3  # Threshold for a long pass based on distance covered
        self.pass_completion_reward = 1.0  # Reward for a successful pass
        self.high_pass_reward = 0.5  # Additional reward for executing a high pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        from_state = self.env.set_state(state)
        return from_state

    def reward(self, reward):
        """
        Implements a reward scheme that reinforces long passing and high passing.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "long_pass_reward": [0.0] * len(reward), "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for player_index, player_reward in enumerate(reward):
            obs = observation[player_index]
            ball_end_pos = obs.get('ball')
            ball_start_pos = obs.get('ball') - obs.get('ball_direction')
            
            # Calculate the distance covered by the ball
            distance = np.linalg.norm(ball_end_pos[:2] - ball_start_pos[:2])

            # Check if the ball is owned by the team and executed as part of a long pass
            if obs.get('ball_owned_team') == 0 and distance >= self.long_pass_threshold:
                if obs.get('ball_direction')[2] > 0:  # Considering vertical component for high pass
                    components["high_pass_reward"][player_index] = self.high_pass_reward
                components["long_pass_reward"][player_index] = self.pass_completion_reward
            
            # Aggregate the rewards
            enhanced_reward = player_reward + components["long_pass_reward"][player_index] + components["high_pass_reward"][player_index]
            reward[player_index] = enhanced_reward

        return reward, components

    def step(self, action):
        """
        Takes a step in the environment with additional computations for rewards based
        on long and high passes.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
