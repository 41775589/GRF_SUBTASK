import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that enhances the training objective focusing on mastering long passes. The reward
    incentivizes making successful long passes under different scenarios, considering the
    distance the ball travels and its precision.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define thresholds for what constitutes a 'long pass'
        self.min_pass_distance = 0.3  # Minimum distance for long pass
        self.last_ball_position = None
        self.precision_factor = 0.05  # Maximum allowed deviation from straight line for precision

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Only consider cases where the current team has ball ownership
            if o['ball_owned_team'] == 1 or o['ball_owned_team'] == 0:
                current_ball_position = np.array(o['ball'][:2])  # Consider XY plane

                if self.last_ball_position is not None:
                    displacement = np.linalg.norm(current_ball_position - self.last_ball_position)

                    # Check if it's a long enough pass
                    if displacement >= self.min_pass_distance:
                        direction_vector = (current_ball_position - self.last_ball_position) / displacement
                        ball_direction = np.array(o['ball_direction'][:2])
                        precision = np.linalg.norm(direction_vector - ball_direction)

                        # Check pass precision
                        if precision <= self.precision_factor:
                            # Reward proportional to the length of the pass
                            components["long_pass_reward"][rew_index] = displacement
                            reward[rew_index] += components["long_pass_reward"][rew_index]

                # Update the ball position tracker
                self.last_ball_position = current_ball_position
            else:
                # Ball is not owned, reset last_ball_position
                self.last_ball_position = None

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
