import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds checkpoint rewards for defensive coordination near the penalty area."""

    def __init__(self, env):
        super().__init__(env)
        self.penalty_area_threshold = 0.4  # Close to the goal area boundary
        self.defensive_coordination_bonus = 0.2
        self.defensive_actions_taken = [False, False]
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_actions_taken = [False, False]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_actions_taken'] = self.defensive_actions_taken
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_actions_taken = from_pickle['defensive_actions_taken']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_coordination": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for agent_index in range(len(reward)):
            o = observation[agent_index]

            # Checking proximity to own goal, factor when close to goal (< penalty_area_threshold)
            if o['left_team'][o['active']][0] < -1 + self.penalty_area_threshold:
                # Check if agent is playing defensively (not owning the ball, near own goal)
                if o['ball_owned_team'] != 0 and not self.defensive_actions_taken[agent_index]:
                    components['defensive_coordination'][agent_index] = self.defensive_coordination_bonus
                    reward[agent_index] += components['defensive_coordination'][agent_index]
                    self.defensive_actions_taken[agent_index] = True

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
