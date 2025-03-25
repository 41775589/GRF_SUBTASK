import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on shooting practice with accuracy and power."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.previous_ball_position = None
        self.shots_on_target = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.previous_ball_position = None
        self.shots_on_target = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shots_on_target'] = self.shots_on_target
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shots_on_target = from_pickle['shots_on_target']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        # If observation is None, do not modify reward
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward}

        for o in observation:
            if 'ball' in o and 'ball_direction' in o:
                # Calculate shooting opportunity
                ball_position = o['ball']
                ball_ownership = o['ball_owned_team']

                # Focus reward on shooting capacities:
                # If the ball has significantly moved towards the opposing goal with a direct line
                if self.previous_ball_position is not None and ball_ownership == 1:  # Assuming attacking direction is from left to right
                    prev_dist_to_goal = self.previous_ball_position[0]
                    current_dist_to_goal = ball_position[0]

                    # Check if the ball is moving towards the goal, and is on the opponent's half
                    if current_dist_to_goal > prev_dist_to_goal and current_dist_to_goal > 0:
                        goal_shot_power = np.linalg.norm(o['ball_direction'][0:2])
                        goal_distance = 1 - current_dist_to_goal

                        # The closer and stronger the shot towards goal, the higher the reward
                        reward += 1.5 * goal_shot_power * goal_distance
                        self.shots_on_target += 1

                self.previous_ball_position = o['ball']

        components['shots_on_target'] = [self.shots_on_target]
        components['ball_position'] = [self.previous_ball_position]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        observations = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in observations:
            for i, action_val in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action_val
        return observation, reward, done, info
