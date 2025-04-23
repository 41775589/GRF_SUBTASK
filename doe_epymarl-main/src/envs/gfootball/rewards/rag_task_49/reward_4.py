import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific shooting reward from central positions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define rewards for different conditions in the central shooting zone
        self.central_shot_power_reward = 0.5
        self.accuracy_bonus = 0.2

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
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_power_reward": [0.0] * len(reward),
                      "accuracy_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, obs in enumerate(observation):
            # Focus on scenarios where the player is near the center with control of the ball aiming at the goal
            if obs['ball_owned_team'] == 1:  # Assuming '1' being the team of agent
                player_x_pos = obs['right_team'][obs['active']][0]
                ball_x_pos = obs['ball'][0]
                
                # Consider only central field shots (around the middle of the x-axis and having control)
                if -0.2 < player_x_pos < 0.2 and abs(ball_x_pos) < 0.1 and obs['ball_owned_player'] == obs['active']:
                    # Encourage more powerful and accurate shots from the central position
                    components['shot_power_reward'][rew_index] = self.central_shot_power_reward
                    reward[rew_index] += self.central_shot_power_reward
                    
                    # Checking if the shot leads to a goal or very close
                    if abs(obs['ball'][1]) < 0.44 and abs(obs['ball'][0]) > 0.9: # assuming near the goal line
                        components['accuracy_bonus'][rew_index] = self.accuracy_bonus
                        reward[rew_index] += self.accuracy_bonus

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
