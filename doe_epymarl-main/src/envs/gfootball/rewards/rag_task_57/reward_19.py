import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for mastering offensive strategies using midfielders and strikers."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define checkpoints for possession and progression for midfielders and successful plays for strikers
        self.possession_rewards = np.zeros((self.env.num_agents,), dtype=float)
        self.progression_rewards = np.zeros((self.env.num_agents,), dtype=float)
        self.finish_play_rewards = np.zeros((self.env.num_agents,), dtype=float)
        # Tune coefficients based on expected influence
        self.possession_coefficient = 0.1
        self.progression_coefficient = 0.3
        self.finish_play_coefficient = 0.6

    def reset(self):
        """Reset the auxiliary rewards and sticky actions on environment reset."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_rewards.fill(0)
        self.progression_rewards.fill(0)
        self.finish_play_rewards.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Save the state of the additional rewards and sticky actions."""
        to_pickle['possession_rewards'] = self.possession_rewards
        to_pickle['progression_rewards'] = self.progression_rewards
        to_pickle['finish_play_rewards'] = self.finish_play_rewards
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Restore the state of the additional rewards and sticky actions."""
        from_pickle = self.env.set_state(state)
        self.possession_rewards = from_pickle['possession_rewards']
        self.progression_rewards = from_pickle['progression_rewards']
        self.finish_play_rewards = from_pickle['finish_play_rewards']
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, reward):
        """Rewards modification to emphasize offensive skills in football."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "possession_reward": self.possession_rewards.tolist(),
            "progression_reward": self.progression_rewards.tolist(),
            "finish_play_reward": self.finish_play_rewards.tolist()
        }

        for i, o in enumerate(observation):
            if o['designated'] == o['active']:  # Check if the agent is controlling the ball
                self.possession_rewards[i] += self.possession_coefficient
            if o['ball_owned_team'] == 1:  # Assuming team 1 is the agent's team
                # Progressive movement towards the opponent's goal
                ball_progression = o['ball'][0] > 0.5  # Halfway across the field
                if ball_progression:
                    self.progression_rewards[i] += self.progression_coefficient
            # Check for goal scoring or critical strike actions
            if o['game_mode'] == 6:  # If the game mode indicates a Penalty or similar
                self.finish_play_rewards[i] += self.finish_play_coefficient
            
            # Aggregate rewards
            reward[i] += (self.possession_rewards[i] +
                          self.progression_rewards[i] +
                          self.finish_play_rewards[i])

        return reward, components

    def step(self, action):
        """Step through the environment and apply rewards."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
