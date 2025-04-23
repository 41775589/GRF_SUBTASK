import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for improving defensive capabilities by training the goalkeeper on shot-stopping and initiating plays, and the defenders on tackling and ball retention."""

    def __init__(self, env):
        """Initialize the environment by setting additional rewards multipliers and the checkpoint counters."""
        super().__init__(env)
        self.goalkeeper_factors = env.observation()['left_team_roles'] == 0  # Identifies the goalkeeper (assuming left_team is being controlled)
        self.shot_stopping_reward = 0.5
        self.init_plays_reward = 0.3
        self.defenders_factors = np.isin(env.observation()['left_team_roles'], [1, 2, 3, 4])  # Centre back, left back, right back, defence midfield
        self.tackling_reward = 0.4
        self.ball_retention_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Reset the sticky action counters and call the default reset method for the environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Maintain the state by storing the necessary data related to the customized reward mechanism."""
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state from the pickled data and the internal state of the environment."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', self.sticky_actions_counter)
        return from_pickle

    def reward(self, reward):
        """Customize the reward function to add rewards for goalkeeper shot-stopping and initiating plays, and for defenders tackling and ball retention."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_shot_stopping": 0.0,
                      "goalkeeper_init_plays": 0.0,
                      "defenders_tackling": 0.0,
                      "defenders_ball_retention": 0.0}

        # Check goalkeeper actions and reward based on goals saved when he is the active player
        if self.goalkeeper_factors[observation['active']]:
            components["goalkeeper_shot_stopping"] += self.shot_stopping_reward * observation['right_team'].shape[0]  # simplistic: number of opponent players towards goal
            components["goalkeeper_init_plays"] += self.init_plays_reward * np.sum(observation['ball_direction'])

        # Check defenders and reward based on tackles made and ball retention when they are the active players
        if self.defenders_factors[observation['active']]:
            components["defenders_tackling"] += self.tackling_reward     # simplistic: adding flat reward for tackle events
            components["defenders_ball_retention"] += self.ball_retention_reward * (1 if observation[
            'ball_owned_team'] == 0 else 0)

        # Aggregate the rewards
        reward += sum(components.values())
        
        return reward, components

    def step(self, action):
        """Wrap the step to include modification in rewards and returning state with detailed info."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return observation, reward, done, info
