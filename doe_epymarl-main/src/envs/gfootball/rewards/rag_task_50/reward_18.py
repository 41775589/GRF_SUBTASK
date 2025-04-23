import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards accurate long passing between specific zones on the field.
    Rewards focus on vision, timing, and precision in the distribution of the ball.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define areas in the field as checkpoints for receiving long passes [x_min, x_max, y_min, y_max]
        self.pass_zones = [
            [-1.0, -0.5, -0.42, 0.42],  # Left half
            [0.5, 1.0, -0.42, 0.42],    # Right half
        ]
        self.last_ball_position = None
        self.pass_reward = 0.5     # Reward for completing a pass to a new zone
        self.max_distance_for_pass = 0.5  # Minimum distance for considering a pass a long pass
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['last_ball_position'] = self.last_ball_position
        return state

    def set_state(self, state):
        self.last_ball_position = state['last_ball_position']
        self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
            
        for rew_index, o in enumerate(observation):
            current_ball_position = o['ball'][:2]  # Get current x, y position of the ball

            if self.last_ball_position is not None:
                # Compute distance the ball has traveled
                distance = np.linalg.norm(current_ball_position - self.last_ball_position)

                if distance >= self.max_distance_for_pass:
                    for zone in self.pass_zones:
                        # Check if the ball has landed in a new pass zone
                        if (zone[0] <= current_ball_position[0] <= zone[1] and
                                zone[2] <= current_ball_position[1] <= zone[3] and
                                not (zone[0] <= self.last_ball_position[0] <= zone[1] and
                                     zone[2] <= self.last_ball_position[1] <= zone[3])):
                            components["pass_reward"][rew_index] = self.pass_reward
                            reward[rew_index] += components["pass_reward"][rew_index]
                
            # Update last ball position for the next step
            self.last_ball_position = current_ball_position

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        info["final_reward"] = sum(reward)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
