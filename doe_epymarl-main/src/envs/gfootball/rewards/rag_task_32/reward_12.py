import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for wingers who execute effective dribbling and crossing."""

    def __init__(self, env):
        super().__init__(env)
        self.winger_positions = {'left': [], 'right': []}
        self.cross_success = False
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.winger_positions = {'left': [], 'right': []}
        self.cross_success = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle.update({'winger_positions': self.winger_positions, 'cross_success': self.cross_success})
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.winger_positions = from_pickle.get('winger_positions', {'left': [], 'right': []})
        self.cross_success = from_pickle.get('cross_success', False)
        return from_pickle

    def reward(self, reward):
        base_reward = reward.copy()
        dribble_reward = [0.0] * len(reward)
        cross_reward = [0.0] * len(reward)

        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {'base_score_reward': base_reward, 'dribble_reward': dribble_reward,
                            'cross_reward': cross_reward}
    
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Assumes the winger is always the last player in the observation list
            if 'right_team_roles' in o and o['right_team_roles'][-1] in {7, 6}:  # winger roles in most setups
                position = o['right_team'][-1]  # position of winger
                if np.linalg.norm(position) > 0.75 and o['ball_owned_player'] == position:
                    dribble_reward[rew_index] = 0.1  # Encourage maintaining control at the wings
                
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                if 'game_mode' in o and o['game_mode'] == 3:  # free kick
                    self.cross_success = True
            
            if self.cross_success:
                cross_reward[rew_index] = 1.0  # Reward successful crossing that leads to free kicks
                self.cross_success = False  # reset for next rewards

        # Update cumulative rewards
        reward = [
            base_reward[i] + dribble_reward[i] + cross_reward[i]
            for i in range(len(reward))
        ]
        
        return reward, {'base_score_reward': base_reward,
                        'dribble_reward': dribble_reward,
                        'cross_reward': cross_reward}

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
