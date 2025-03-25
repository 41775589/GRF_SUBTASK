import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for successful sliding tackles by defensive players under high-pressure situations.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """
        Reset the environment and the sticky actions counter at the beginning of each episode.
        """
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """
        Allows the environment state to be saved, including the wrapper's specific state.
        """
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        """
        Allows the environment state to be restored, including the wrapper's specific state.
        """
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions'], dtype=int)
        return from_pickle

    def reward(self, reward):
        """
        Adjust the original reward based on the timely and successful execution of sliding tackles.
        The reward itself is based on whether the action was a sliding tackle by a defensive player (roles 1, 3, or 4).
        Extra reward is given if this occurs when the opposing team is attacking (ball position in the own half).
        """
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'tackle_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_role = o['right_team_roles'][o['active']] if o['active'] >= 0 else -1
            ball_position = o['ball'][0]  # Ball's x-coordinate on the field
            
            # Defensive roles are 1 (centre back), 3 (right back), and 4 (defence midfield)
            if active_player_role in [1, 3, 4] and ball_position <= 0:
                # Check if the agent performed a slide tackle
                if o['sticky_actions'][6]:  # action 6 corresponds to sliding
                    components['tackle_reward'][rew_index] = 0.5  # custom reward for successful sliding tackles
                    reward[rew_index] += components['tackle_reward'][rew_index]
        
        return reward, components

    def step(self, action):
        """
        This method should not be changed when adding custom reward logic.
        It ensures that the reward modifications are correctly applied and returned.
        """
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        self.sticky_actions_counter.fill(0)
        for agent_obs in self.env.unwrapped.observation():
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f'sticky_actions_{i}'] = action
        return observation, reward, done, info
