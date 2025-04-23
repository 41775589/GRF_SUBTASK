import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward for goalkeeper coordination during high-pressure scenarios."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clear_attempts = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clear_attempts = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['clear_attempts'] = self.clear_attempts
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.clear_attempts = from_pickle.get('clear_attempts', 0)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "goalkeeper_support_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        for i, o in enumerate(observation):
            goalkeeper_support_reward = 0
            ball_position = o['ball']
            ball_owned_team = o['ball_owned_team']
            active_player_role = o['left_team_roles'][o['active']] if ball_owned_team == 0 else o['right_team_roles'][o['active']]
            
            # Encourage clearing the ball away from the goal area under pressure
            prob_clearing = np.random.uniform(0, 1)
            clear_threshold = 0.95  # Simulated probability threshold for effective clearing under pressure
            if active_player_role == 0 and prob_clearing > clear_threshold:
                goalkeeper_support_reward += 0.5
                self.clear_attempts += 1

            components["goalkeeper_support_reward"][i] = goalkeeper_support_reward
            reward[i] += goalkeeper_support_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
            
        # Track actions considered 'sticky' for later analytics or monitoring
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
