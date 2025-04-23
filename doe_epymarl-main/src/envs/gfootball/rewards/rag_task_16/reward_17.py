import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward based on executing high passes with precision.
    It focuses on trajectory control, power assessment, and situational application of high passes.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Parameters for high pass control
        self.passing_threshold_height = 0.15  # The minimum z height to qualify as a high pass
        self.precision_bonus = 0.1  # Additional reward for precision in high passes

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        # Initialize reward components
        components = {"base_score_reward": reward.copy(),
                      "high_pass_skill": [0.0, 0.0]}
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_z = o['ball'][2]  # Z coordinate of the ball
            ball_direction = o['ball_direction']
            ball_velocity_z = ball_direction[2]

            # Assessing the z height of the ball
            if ball_z > self.passing_threshold_height and ball_velocity_z > 0:
                # Check if the ball is moving upwards and above the threshold height to consider it a high pass
                
                # Calculating precision based on current game state
                if o['ball_owned_team'] == 1:  # If the ball is owned by right team
                    teammate_positions = [pos for pos, active in zip(o['right_team'], o['right_team_active']) if active]
                else:
                    teammate_positions = [pos for pos, active in zip(o['left_team'], o['left_team_active']) if active]

                # Find players around the ball landing position (projected)
                projected_ball_landing = np.array([o['ball'][0] + ball_direction[0], o['ball'][1] + ball_direction[1]])
                for pos in teammate_positions:
                    dist = np.linalg.norm(pos - projected_ball_landing[:2])
                    if dist < 0.1:  # Assuming a small range as precise pass
                        components['high_pass_skill'][rew_index] += self.precision_bonus
                        reward[rew_index] += components['high_pass_skill'][rew_index]

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
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
