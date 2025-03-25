import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adjusts the reward function to emphasize successful, powerful shots.
    It distinguishes between shots made under pressure and those made without immediate threats.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the wrapper state for a new episode."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Include the state of the wrapper in the environment's state."""
        to_pickle['CheckpointRewardWrapper'] = {"StickyActionsCounter": self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the wrapper from the environment's state."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['StickyActionsCounter'])
        return from_pickle

    def reward(self, reward):
        """
        Adjust the base reward given by the environment based on the shot's quality and pressure distinction.
        
        Reward logic:
        - Base reward is modified by checking if the action taken was a successful shot.
        - Additional rewards are given for shots taken under pressure.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pressure_shot_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            pressure_factor = self.calculate_pressure(o['opponents_dist'], o['ball_position'])
            if o['action_taken'] == 'shot':
                if o['shot_success']:
                    reward[rew_index] += 1  # Boost for scoring
                    if pressure_factor > 0.7:
                        # Higher reward if the shot was made under high pressure
                        components['pressure_shot_reward'][rew_index] = 1.0
                        reward[rew_index] += components['pressure_shot_reward'][rew_index]
        
        return reward, components

    def calculate_pressure(self, opponents_dists, ball_position):
        """
        Calculate pressure based on closeness of opponents and the position of the ball.
        Higher values mean more pressure.
        """
        dists = np.array(opponents_dists)
        return np.exp(-np.mean(dists) / (0.1 if ball_position > 0.5 else 0.2))

    def step(self, action):
        """Collect data, preprocess it, step the environment, and apply rewards."""
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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
