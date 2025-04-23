import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides a specific reward function aimed at trainng a goalkeeper in shot stopping, 
    quick decision-making for ball distribution under pressure, and effective communication with defenders.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize parameters for custom rewards
        self.shot_stop_reward = 2.0
        self.distribution_reward = 1.0
        self.communication_reward = 0.5
        
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "shot_stop_reward": np.zeros_like(reward),
            "distribution_reward": np.zeros_like(reward),
            "communication_reward": np.zeros_like(reward)
        }
        
        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            goalie_index = np.where(o['left_team_roles'] == 0)[0]
            
            if o['active'] in goalie_index:
                # Assumption that 'ball_owned_player' will be the index of the player from the respective team.
                # Increase reward for stopping shots (ball coming towards goal with high speed and close to goal)
                if o['ball_owned_team'] == 1 and np.linalg.norm(o['ball_direction'][:2]) > 0.1:
                    if np.abs(o['ball'][0] - (-1)) < 0.2 and np.abs(o['ball'][1]) < 0.07:
                        components['shot_stop_reward'][rew_index] = self.shot_stop_reward

                # Reward for distributing the ball to defenders
                if 'action' in o and o['action'] == 'long_pass' and o['ball_owned_team'] == 0:
                    components['distribution_reward'][rew_index] = self.distribution_reward

                # Reward for staying positioned correctly relative to defenders
                defenders = [i for i in range(len(o['left_team'])) if o['left_team_roles'][i] in (1, 2, 3)]
                avg_defender_x = np.mean([o['left_team'][i][0] for i in defenders])
                if np.abs(avg_defender_x - o['left_team'][o['active']][0]) < 0.1:  # Reasonable threshold for alignment in x-axis
                    components['communication_reward'][rew_index] = self.communication_reward
            
            total_reward = np.sum([components[key][rew_index] for key in components])
            reward[rew_index] += total_reward
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
