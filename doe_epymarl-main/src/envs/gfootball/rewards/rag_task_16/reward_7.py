import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A Reward Wrapper that focuses on technically skilled high passes. It rewards precision,
    suitable power application and correct trajectory planning.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, from_pickle):
        state = self.env.set_state(from_pickle)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return state

    def reward(self, reward):
        """
        Modify the rewards based on high pass execution.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for index, obs in enumerate(observation):
            components["high_pass_reward"][index] = 0

            # Basic requirements for high pass consideration
            if obs['game_mode'] in [0, 5]:  # Normal Play and Throw In only
                # Check for high pass condition, simplistic approximation using ball direction and speed
                ball_z_velocity = obs['ball'][2] # assuming this index corresponds to z-axis velocity
                if ball_z_velocity > 0.1:        # Just a threshold to detect "high" movement
                    # Reward for successful high pass execution
                    proximity_to_goal = abs(obs['ball'][0] - 1)  # Assuming ball[0] is the x coordinate to the right goal
                    if proximity_to_goal < 0.2 and obs['ball_owned_team'] == 0:
                        components["high_pass_reward"][index] = 1.0  # Strong reward if close to opponent's goal
                    else:
                        components["high_pass_reward"][index] = 0.5  # Lesser reward for just the attempt

            # Combine rewards, giving a partial boost for high pass skills
            reward[index] += 0.5 * components["high_pass_reward"][index]

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
