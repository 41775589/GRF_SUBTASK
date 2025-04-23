import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a scenario-focused reward for shooting and passing precision, decision-making, and strategic positioning."""

    def __init__(self, env):
        super().__init__(env)
        self.pass_stretch_weight = 0.2   # Weight for rewards related to accurate long passes
        self.shoot_accuracy_weight = 0.3  # Weight for rewards for shots on goal
        self.positioning_weight = 0.5     # Weight for strategic positioning
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Get the state of the environment for serialization."""
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """Set the state of the environment from deserialization."""
        return self.env.set_state(state)

    def reward(self, reward):
        """Modify the rewards based on specific gaming scenarios focusing on passing, shooting, and positioning."""
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "pass_stretch_reward": [0.0] * len(reward),
            "shoot_accuracy_reward": [0.0] * len(reward),
            "positioning_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for i in range(len(reward)):
            o = observation[i]

            # Reward for successful long passes
            if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball_direction'][:2]) > 0.1:
                components["pass_stretch_reward"][i] = self.pass_stretch_weight

            # Reward for accurate shots towards the goal
            if o['game_mode'] == 6 and o['ball_owned_team'] == 1:  # In a shooting scenario
                if np.abs(o['ball'][0]) > 0.7 and np.abs(o['ball'][1]) < 0.1:  # Ball close to the goal area
                    components["shoot_accuracy_reward"][i] = self.shoot_accuracy_weight

            # Reward for good strategic positioning
            if 'right_team_roles' in o and o['right_team_roles'][o['active']] == 9:  # if agent is a striker
                distance_to_goal = np.linalg.norm(o['right_team'][o['active']] - [1, 0])  # Distance to opponent goal
                if distance_to_goal < 0.3:
                    components["positioning_reward"][i] = self.positioning_weight

            # Total reward computation
            total_rewards = (components["base_score_reward"][i] +
                             components["pass_stretch_reward"][i] +
                             components["shoot_accuracy_reward"][i] +
                             components["positioning_reward"][i])

            reward[i] = total_rewards

        return reward, components

    def step(self, action):
        """Processes the environment's step including tracking and altering rewards based on defined parameters."""
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
