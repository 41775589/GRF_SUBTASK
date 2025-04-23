import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to enhance training by rewarding successful tackles and regaining possession."""
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define extra parameters for rewarding tackles and successful possession regains
        self.possession_reward = 0.05
        self.tackle_reward = 0.1
        self.prev_ball_owned_team = None

    def reset(self):
        """Resets the sticky actions counter and environment."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_owned_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        """Serializes the current state of the wrapper including rewards."""
        state = super(CheckpointRewardWrapper, self).get_state(to_pickle)
        state['prev_ball_owned_team'] = self.prev_ball_owned_team
        return state

    def set_state(self, state):
        """Deserializes the state of the wrapper."""
        from_pickle = super(CheckpointRewardWrapper, self).set_state(state)
        self.prev_ball_owned_team = state['prev_ball_owned_team']
        return from_pickle

    def reward(self, reward):
        """Calculate additional rewards for successful tackles and possession regaining."""
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(), 'tackle_bonus': [0.0] * len(reward)}

        if observation is None:
            return reward, components
       
        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            current_ball_team = o['ball_owned_team']

            # Reward for regaining possession
            if self.prev_ball_owned_team is not None and current_ball_team == 0:
                if self.prev_ball_owned_team == 1:
                    components['tackle_bonus'][i] = self.tackle_reward
                    reward[i] += components['tackle_bonus'][i]

            # Update the previous ball ownership to the current state
            self.prev_ball_owned_team = current_ball_owned_team

        return reward, components

    def step(self, action):
        """Overriding the step function to include custom reward adjustments."""
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
