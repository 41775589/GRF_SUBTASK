import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for the 'stopper' role emphasizing man-marking, shot-blocking and stalling moves."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Initialize variables to track the environment state and sticky actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Tuning constants for rewards
        self.man_marking_reward = 0.05
        self.shot_blocking_reward = 0.2
        self.stalling_reward = 0.01

    def reset(self):
        # Reset sticky actions counter when the environment is reset
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
                      "man_marking_reward": [0.0] * len(reward),
                      "shot_blocking_reward": [0.0] * len(reward),
                      "stalling_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            current_position = o['right_team'][o['active']]
            ball_position = o['ball'][:2]
            ball_possession = o['ball_owned_team']
            distance_to_ball = np.linalg.norm(ball_position - current_position)

            if ball_possession == 1:  # If ball is owned by the opponent
                # Reward for being close to the player who has the ball
                if distance_to_ball < 0.05:
                    components["man_marking_reward"][rew_index] += self.man_marking_reward

                # 'block shot': if close to trajectory of ball towards the goal
                goal_position = [1, 0]
                if abs(current_position[1] - goal_position[1]) < 0.1:
                    components["shot_blocking_reward"][rew_index] += self.shot_blocking_reward

                # Reward for stalling opponent's forward motion
                if o['right_team_direction'][o['active']][0] < 0:  # The player's x-direction movement towards opponent's goal
                    components["stalling_reward"][rew_index] += self.stalling_reward

            # Update the overall reward with new components
            reward[rew_index] += (components["man_marking_reward"][rew_index] +
                                  components["shot_blocking_reward"][rew_index] +
                                  components["stalling_reward"][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        # Update the sticky actions tracking
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
