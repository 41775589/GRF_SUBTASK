import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward specifically tailored for coordinating two agents in defensive scenarios."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_synergy_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components
        
        # Implement a team defensive coverage reward
        for i in range(len(observation)):
            o = observation[i]
            if o['ball_owned_team'] == 1:  # ball owned by opponent team
                team_positions = o['left_team']
                player_position = team_positions[o['active']]
                ball_position = o['ball'][:2]  # Ignore z-coordinate
                
                # Calculate distances from players to the ball
                distances = np.linalg.norm(team_positions - ball_position, axis=1)
                team_proximity_to_ball = np.sort(distances)[:2]  # Closest two defenders
                
                # Encourage the two agents to collaboratively reduce the distance to the ball
                if i < 2:  # Assuming the two agents are the first two indexes
                    components["defensive_synergy_reward"][i] = -np.sum(team_proximity_to_ball)
                    reward[i] += components["defensive_synergy_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
