import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for demonstrating offensive skills in football tactics like passing, shooting, and dribbling."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Initialize sticky actions tracking

    def reset(self):
        """Resets the environment and sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """
        Enhances the base reward by adding a tactical reward focusing on actions like Short Pass, Long Pass,
        Shot, Dribble, and Sprint based on current game state and player's actions.
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "tactical_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, rew in enumerate(reward):
            o = observation[rew_index]
            if o is None:
                continue
            
            # Encourage actions that are tactically advantageous
            active_actions = o.get('sticky_actions', np.zeros(10))
            has_ball = o.get('ball_owned_team', -1) == 1  # Assuming our team is team 1
            
            # Points for controlling and manipulating the game through dribbles and passes when in possession
            if has_ball:
                tactical_points = 0.01 * active_actions[8]  # Sprint
                tactical_points += 0.02 * active_actions[9]  # Dribble
                tactical_points += 0.05 * (active_actions[3] + active_actions[4]) # Short and long passes

                # Special reward for attempting shots on goal
                if o.get('game_mode', 0) in [0, 5]:  # Normal play or Penalty Kick
                    tactical_points += 0.10 * active_actions[5]  # Shot action

                components["tactical_reward"][rew_index] += tactical_points
                reward[rew_index] += components["tactical_reward"][rew_index]

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

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)
