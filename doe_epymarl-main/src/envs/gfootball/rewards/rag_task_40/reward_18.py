import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward to enhance agents' defensive skills
    toward direct attacks.
    This reward encourages maintaining defensive positions and intercepting the ball.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initializes intercept and position checkpoints
        self.intercept_checkpoints = [False, False]  # For intercepting the ball
        self.position_checkpoints = [None, None]  # Initially no defined positions
        self.position_rewards = [0.0, 0.0]
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the episode and clear checkpoints.
        """
        self.intercept_checkpoints = [False, False]
        self.position_checkpoints = [None, None]
        self.position_rewards = [0.0, 0.0]
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Returns the state with checkpoints included.
        """
        to_pickle['CheckpointRewardWrapper'] = (self.intercept_checkpoints,
                                                self.position_checkpoints,
                                                self.position_rewards)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Set the state with checkpoints.
        """
        from_pickle = self.env.set_state(state)
        self.intercept_checkpoints, self.position_checkpoints, self.position_rewards = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        """
        Compute the reward based on defense effectiveness: maintaining position and intercepting.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "intercept_reward": [0.0] * len(reward),
                      "position_reward": self.position_rewards.copy()}

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]
            if o['ball_owned_team'] == 0 and not self.intercept_checkpoints[i]:
                self.intercept_checkpoints[i] = True
                components["intercept_reward"][i] = 0.2  # Reward for intercepting the ball

            # Calculate player-baseline distance maintenance (strategy positioning)
            # Assuming 'defense_line' is a heuristic metric provided, e.g., mean(y-coord of defense players)
            defense_line_y = np.mean([p[1] for p in o['left_team']])
            player_y = o['left_team'][i][1]
            if self.position_checkpoints[i] is None:
                self.position_checkpoints[i] = player_y

            # Still near the original positional line?
            if abs(self.position_checkpoints[i] - player_y) < 0.1:
                self.position_rewards[i] += 0.01
                components["position_reward"][i] = 0.01
            else:
                self.position_rewards[i] = 0.0

            reward[i] += components["intercept_reward"][i] + components["position_reward"][i]

        return reward, components

    def step(self, action):
        """
        Takes a step in the environment and processes the associated rewards.
        """
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
