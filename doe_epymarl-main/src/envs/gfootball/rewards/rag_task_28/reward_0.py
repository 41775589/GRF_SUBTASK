import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific dribbling skill enhancement reward focused on face-to-face interactions
    with the goalkeeper, emphasizing quick feints, sudden direction changes, and maintaining ball control under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_pos = np.array([0.0, 0.0])
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_pos = np.array([0.0, 0.0])
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['previous_ball_pos'] = self.previous_ball_pos
        return state

    def set_state(self, state):
        self.previous_ball_pos = state['previous_ball_pos']
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "dribble_skill_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for i in range(len(reward)):
            obs = observation[i]
            ball_owned = obs['ball_owned_player'] == obs['active']
            ball_controlled = obs['ball_owned_team'] == 1  # Assuming team 1 is the agent's team

            if ball_owned and ball_controlled:
                current_ball_pos = np.array(obs['ball'][:2])  # x, y positions only
                ball_distance_to_goal = np.abs(current_ball_pos[0] - 1)  # Distance to opponent's goal on x-axis

                # Encourage maintaining ball control
                if np.any(np.abs(self.previous_ball_pos - current_ball_pos) > 0.01):
                    components["dribble_skill_reward"][i] += 0.05

                # Reward for approaching goal with controlled dribbling
                if ball_distance_to_goal < np.abs(self.previous_ball_pos[0] - 1):
                    components["dribble_skill_reward"][i] += 0.1

                # Update ball position
                self.previous_ball_pos = current_ball_pos
        
        # Aggregate all components into final reward
        for i in range(len(reward)):
            reward[i] += components["dribble_skill_reward"][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()

        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                # Monitoring sticky actions such as sprint and dribble
                self.sticky_actions_counter['sprint'] = agent_obs['sticky_actions'][8]
                self.sticky_actions_counter['dribble'] = agent_obs['sticky_actions'][9]
        return observation, reward, done, info
