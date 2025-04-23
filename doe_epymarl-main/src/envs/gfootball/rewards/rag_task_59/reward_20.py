import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on goalkeeper coordination and clearing the ball."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_coordination_reward = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        # Base score reward is captured from the original reward obtained
        components = {"base_score_reward": reward.copy()}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components
        
        # Initialize the custom reward component
        components["goalkeeper_coordination"] = [0.0] * len(reward)

        for idx, o in enumerate(observation):
            # Identify the goalkeeper (role index 0)
            is_goalkeeper = o['left_team_roles'][o['active']] == 0
            ball_owned_by_team = o['ball_owned_team'] == 0

            if is_goalkeeper and ball_owned_by_team:
                # Calculate the distance from the ball to the nearest opponent
                ball_position = o['ball'][:2]
                opponent_positions = o['right_team']

                # Calculate distances to all opponents
                distances = np.linalg.norm(opponent_positions - ball_position, axis=1)
                # Consider it high-pressure if any opponents are closer than 0.1 units
                if np.any(distances < 0.1):
                    # Reward the goalkeeper for having the ball under pressure
                    components["goalkeeper_coordination"][idx] = self.goalkeeper_coordination_reward
                    reward[idx] += components["goalkeeper_coordination"][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Include reward components in the info dictionary for monitoring
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Track sticky actions
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
