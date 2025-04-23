import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes mastering midfield dynamics, enhancing coordination under pressure,
    and strategic repositioning during offensive and defensive transitions.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.midfield_x_threshold = 0.0  # the x coordinate defining the midfield line
        self.possession_change_reward = 0.5
        self.midfield_control_reward = 0.1
        self.defensive_to_offensive_transition_reward = 0.3
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_owner = None
        self.last_game_mode = 0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_owner = None
        self.last_game_mode = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_owner'] = self.last_ball_owner
        to_pickle['last_game_mode'] = self.last_game_mode
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_owner = from_pickle['last_ball_owner']
        self.last_game_mode = from_pickle['last_game_mode']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": list(reward).copy()}

        # Initialize component scores
        components["possession_change_reward"] = [0.0] * len(reward)
        components["midfield_control_reward"] = [0.0] * len(reward)
        components["transition_reward"] = [0.0] * len(reward)

        if observation is None:
            return list(reward), components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward changes in possession if it leads to positive play
            if o['ball_owned_team'] != self.last_ball_owner:
                if o['ball_owned_team'] == 1:  # Assuming 1 is our team
                    components["possession_change_reward"][rew_index] = self.possession_change_reward
                    reward[rew_index] += components["possession_change_reward"][rew_index]
                self.last_ball_owner = o['ball_owned_team']

            # Reward maintaining the ball in the midfield area to control game flow
            if abs(o['ball'][0]) <= self.midfield_x_threshold and o['ball_owned_team'] == 1:
                components["midfield_control_reward"][rew_index] = self.midfield_control_reward
                reward[rew_index] += components["midfield_control_reward"][rew_index]

            # Reward transitions from defense to offense
            if self.last_game_mode in {2, 3, 4, 5, 6} and o['game_mode'] == 0 and o['ball_owned_team'] == 1:
                components["transition_reward"][rew_index] = self.defensive_to_offensive_transition_reward
                reward[rew_index] += components["transition_reward"][rew_index]

            self.last_game_mode = o['game_mode']

        return list(reward), components

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
