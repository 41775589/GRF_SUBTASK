import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds specialized rewards for offensive maneuvers during various
    game phases to enhance quick attack response.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Rewards based on distance to the opponent's goal, offensive maneuvers and game phases
        self.offensive_maneuver_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        """
        Modify rewards with respect to offensive strategy, tracking ball control
        closer to opponent's goal and during specific game phases.
        """
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "offensive_maneuver_reward": [0.0] * len(reward)
        }

        # Ensure the observation is not empty
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward for ball control near the opponent's goal
            ball_position = o['ball'][0]  # x-coordinate of the ball
            if o['ball_owned_team'] == 1:  # If the right team owns the ball which is the opponent
                distance_to_goal = 1 - ball_position  # Distance from right goal is (1 - current position)
            else:
                continue

            # Check for offensive maneuvers during game modes
            game_mode = o['game_mode']
            if game_mode in [0, 2, 4]:  # Normal, GoalKick, or Corner
                if ball_position > 0.5:  # Ball is on the opponent's side
                    components['offensive_maneuver_reward'][rew_index] = self.offensive_maneuver_reward

            # Update the reward
            reward[rew_index] += components['offensive_maneuver_reward'][rew_index]

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
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
