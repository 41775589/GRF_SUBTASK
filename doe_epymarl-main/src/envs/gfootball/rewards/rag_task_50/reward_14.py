import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A custom reward wrapper for specializing in executing long passes 
    across different areas of the field, focusing on vision, timing, 
    and precision in ball distribution.
    """

    def __init__(self, env):
        super().__init__(env)
        self.previous_ball_pos = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_pos = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'previous_ball_pos': self.previous_ball_pos
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_pos = from_pickle['CheckpointRewardWrapper'].get('previous_ball_pos')
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': reward.copy()}

        components = {'base_score_reward': reward.copy(), 'long_pass_reward': [0.0] * len(reward)}

        for idx in range(len(reward)):
            current_obs = observation[idx]
            if self.previous_ball_pos is not None and current_obs['ball_owned_team'] == 1:
                ball_pos = np.array(current_obs['ball'])
                prev_ball_pos = np.array(self.previous_ball_pos)

                # Calculating the distance the ball has moved in Euclidean terms.
                ball_dist_moved = np.linalg.norm(ball_pos - prev_ball_pos)
                # Reward based on the distance the ball was passed over, emphasizing longer passes.
                if ball_dist_moved > 0.3:  # This threshold can be changed to suit what is considered a "long pass".
                    components['long_pass_reward'][idx] = ball_dist_moved  # Reward proportional to distance.

                # Updating the reward for the agent.
                reward[idx] += components['long_pass_reward'][idx]

            # Update previous ball position
            self.previous_ball_pos = current_obs['ball']

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
