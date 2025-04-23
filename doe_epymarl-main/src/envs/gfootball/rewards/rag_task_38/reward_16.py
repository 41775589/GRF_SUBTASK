import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for initiating counterattacks through accurate long passes and quick transitions from defense to attack."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_position_last_step = None
        self.long_pass_threshold = 0.5  # Define the minimum distance for a pass to be considered long.
        self.counterattack_reward = 1.0  # Reward weight for a successful counterattack initiation.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_position_last_step = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'ball_position_last_step': self.ball_position_last_step}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.ball_position_last_step = from_pickle['CheckpointRewardWrapper'].get('ball_position_last_step', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "counterattack_bonus": [0.0] * len(reward)
        }
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            current_observation = observation[rew_index]
            current_ball_position = current_observation['ball'][:2]  # [x, y] position of the ball.

            if self.ball_position_last_step is not None:
                distance_moved = np.linalg.norm(current_ball_position - self.ball_position_last_step)
                
                # Check if a long pass was made based on distance the ball traveled and if the ball has switched teams
                if distance_moved > self.long_pass_threshold and current_observation['ball_owned_team'] == 1:
                    # A long pass is often part of a counterattack strategy.
                    components["counterattack_bonus"][rew_index] = self.counterattack_reward
                    reward[rew_index] += components["counterattack_bonus"][rew_index]

            # Updating the ball position for the next step.
            self.ball_position_last_step = current_ball_position

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
