import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for accuracy and power in shots from central field positions."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shooting_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            obs = observation[rew_index]
            ball_pos = obs['ball'][:2]  # only consider x, y

            # Check if player is in the central field area and has the ball
            if (-0.2 <= ball_pos[0] <= 0.2) and obs['ball_owned_team'] == 0:
                # Compute shooting effectiveness based on ball position and direction
                goal_dir = np.array([1, 0])  # Direction towards the goal for the left team
                ball_direction = obs['ball_direction'][:2]
                direction_cosine = np.dot(goal_dir, ball_direction) / (np.linalg.norm(goal_dir) * np.linalg.norm(ball_direction))

                # The closer the direction cosine to 1, the more aligned the shot, hence higher reward
                if direction_cosine > 0.5:  # Threshold for considering a valid direction
                    power = np.linalg.norm(ball_direction)  # Use norm as a proxy for power
                    components["shooting_reward"][rew_index] = direction_cosine * power  # Reward combines both accuracy and power

                # Integrate this reward into the total reward
                reward[rew_index] += components["shooting_reward"][rew_index]

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
