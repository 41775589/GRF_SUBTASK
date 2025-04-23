import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes long-range shots, especially outside the penalty box."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define distance thresholds for long shots outside the penalty area.
        self.long_shot_threshold = 0.6  # Approximately outside the penalty box
        self.reward_for_long_shot = 0.3   # Additional reward for long shots
        self.shot_attempted = False

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.shot_attempted = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['shot_attempted'] = self.shot_attempted
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.shot_attempted = from_pickle.get('shot_attempted', False)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "long_shot_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components
        
        for i in range(len(reward)):
            player_obs = observation[i]
            ball_pos_x = player_obs['ball'][0]
            
            # Check if the ball is in the right team’s control near the opponent’s goal outside the penalty box
            if player_obs['ball_owned_team'] == 1 and abs(ball_pos_x) > self.long_shot_threshold:
                # Check if a shot was attempted
                action_shot = player_obs['sticky_actions'][0] # assuming 0 index is the 'shoot' action
                if action_shot == 1:
                    self.shot_attempted = True
                    components['long_shot_reward'][i] = self.reward_for_long_shot
                    reward[i] += components['long_shot_reward'][i]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, act in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = act
        return observation, reward, done, info
