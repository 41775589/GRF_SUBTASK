import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies reward functions to focus more on effective offensive plays."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_reward_coeff = 0.3
        self.shot_reward_coeff = 3.0
        self.dribble_reward_coeff = 0.1
        self.possession_reward_coeff = 0.2

    def reset(self):
        """Reset and return the initial observation from the environment."""
        return self.env.reset()

    def get_state(self, to_pickle):
        """Store the state of this reward wrapper along with the environment's state."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state from the given state object."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the reward based on the offensive capabilities demonstrated by the agents."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "pass_reward": np.zeros(len(reward)),
            "shot_reward": np.zeros(len(reward)),
            "dribble_reward": np.zeros(len(reward)),
            "possession_reward": np.zeros(len(reward))
        }

        for player_index, obs in enumerate(observation):
            # Improve shot reward only when game mode indicates a potential shot opportunity
            if obs['game_mode'] in [3]:  # Shot modes, like penalty kicks etc.
                if obs['ball_owned_player'] == obs['active']:
                    components["shot_reward"][player_index] = self.shot_reward_coeff
                    reward[player_index] += components["shot_reward"][player_index]

            # Reward player for passes only when the player passes successfully
            if 'sticky_actions' in obs and np.any(obs['sticky_actions'][1:3]):
                components["pass_reward"][player_index] = self.pass_reward_coeff
                reward[player_index] += components["pass_reward"][player_index]

            # Reward for dribbling based on ball possession change
            if obs['ball_owned_team'] == (0 if obs['left_team_active'][obs['active']] else 1):
                components["dribble_reward"][player_index] = self.dribble_reward_coeff
                reward[player_index] += components["dribble_reward"][player_index]

            # Reward simply for having the ball, encourages maintaining possession
            if obs['ball_owned_player'] == obs['active']:
                components["possession_reward"][player_index] = self.possession_reward_coeff
                reward[player_index] += components["possession_reward"][player_index]

        return reward, components

    def step(self, action):
        """Step the environment by applying action, modifying the reward, and returning observations."""
        observation, reward, done, info = self.env.step(action)
        # Assign the new reward based on offensive play-focused reward function
        reward, components = self.reward(reward)
        # Add the final reward and components to the info for more detailed analysis
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        return observation, reward, done, info
