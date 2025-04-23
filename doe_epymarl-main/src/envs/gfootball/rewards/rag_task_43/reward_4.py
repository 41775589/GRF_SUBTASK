import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for defensive play and efficient counterattacks,
    aiming to enhance defensive strategy by improving positional awareness,
    responsiveness, and effective transition into counterattacks.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        # No specific state loading required for this wrapper
        return from_pickle

    def reward(self, reward):
        # Initialization of the component rewards dictionary
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "counterattack_reward": [0.0] * len(reward)}

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        # Applying defensive and counterattack rewards
        for i, obs in enumerate(observation):
            if obs['ball_owned_team'] == 1:  # If the opposing team owns the ball
                distance_to_ball = np.linalg.norm(
                    obs['left_team'][obs['active']] - obs['ball'][:2])
                # Encourage closer positioning to the ball
                if distance_to_ball < 0.1:
                    components["defensive_reward"][i] += 0.05

            if obs['game_mode'] == 0 and obs['ball_owned_team'] == 0:  # Normal game mode and our team has the ball
                # Reward pushing forward quickly from defense to attack
                x_position = obs['left_team'][obs['active']][0]
                if x_position > 0.75:  # Advances into the opponent's half
                    components["counterattack_reward"][i] += 0.1

        # Update the reward based on the computed components
        for i in range(len(reward)):
            reward[i] += (components["defensive_reward"][i] +
                          components["counterattack_reward"][i])
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
