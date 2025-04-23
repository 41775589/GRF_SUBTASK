import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for mastering long passes in football.
    It encourages accuracy and understanding of ball dynamics over different lengths.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position = np.zeros(3)  # To track the last position of the ball
        self.pass_accuracy_reward = 0.2  # Reward for completing a pass within close proximity to a teammate

    def reset(self):
        self.last_ball_position = np.zeros(3)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self.last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        if 'last_ball_position' in from_pickle:
            self.last_ball_position = from_pickle['last_ball_position']
        return from_pickle

    def reward(self, reward):
        # Retrieve the latest state of the football match
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "precision_pass_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components
        
        # Loop through observations for both agents
        for rew_idx, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                # Calculate the distance the ball traveled since the last step if the same team still owns it
                ball_travel_distance = np.linalg.norm(o['ball'][:2] - self.last_ball_position[:2])
                # Check if the ball distance shows a plausible long pass (significant movement without scoring a goal)
                if 0.3 < ball_travel_distance < 0.7 and reward[rew_idx] == 0:
                    # Evaluate the accuracy by checking if a teammate is near the new position
                    team_positions = o['left_team'] if o['ball_owned_team'] == 0 else o['right_team']
                    distances_to_teammates = [np.linalg.norm(o['ball'][:2] - teammate_pos) for teammate_pos in team_positions]
                    if any(dist < 0.1 for dist in distances_to_teammates):  # Considered accurate if within a 0.1 radius
                        components["precision_pass_reward"][rew_idx] = self.pass_accuracy_reward
                        reward[rew_idx] += components["precision_pass_reward"][rew_idx]

            self.last_ball_position = o['ball']  # Update last ball position for next step

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
