import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards for collaborative plays leading to scoring opportunities."""

    def __init__(self, env):
        super().__init__(env)
        self.collaboration_bonus = 0.05
        self.passing_distance_threshold = 0.3  # Arbitrary threshold to consider a significant pass
        self.previous_ball_owner = -1
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = -1
        return self.env.reset()

    def get_state(self, to_pickle):
        state_pickle = self.env.get_state(to_pickle)
        state_pickle['previous_ball_owner'] = self.previous_ball_owner
        return state_pickle

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner = from_pickle['previous_ball_owner']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {
            "base_score_reward": reward.copy(),
            "collaboration_bonus": [0.0] * len(reward)
        }

        if observation['ball_owned_team'] == 1:  # Check if the right team owns the ball
            current_ball_owner = observation['ball_owned_player']
            if current_ball_owner != self.previous_ball_owner and self.previous_ball_owner != -1:
                
                # Calculate distance between previous and current owner
                prev_position = observation['right_team'][self.previous_ball_owner]
                curr_position = observation['right_team'][current_ball_owner]
                distance = np.sqrt((prev_position[0] - curr_position[0])**2 + (prev_position[1] - curr_position[1])**2)
                
                if distance > self.passing_distance_threshold:
                    # Approve collaboration reward
                    for i, _ in enumerate(reward):
                        components["collaboration_bonus"][i] = self.collaboration_bonus
                        reward[i] += self.collaboration_bonus

            # Update the previous ball owner
            self.previous_ball_owner = current_ball_owner
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
