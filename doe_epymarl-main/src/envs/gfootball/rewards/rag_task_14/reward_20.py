import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper focusing on the role of a 'sweeper' in a football game."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        """Resets the environment and the sticky actions counter."""
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        """Gets the state of the environment, enhanced with the current wrapper's state."""
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        """Sets the state of the environment from serialized data."""
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        """Computes the additional reward for sweeper actions such as clearing balls and tackles."""
        observation = self.env.unwrapped.observation()
        additional_rewards = {"base_score_reward": reward.copy(),
                              "clearance_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, {}

        assert len(reward) == len(observation)

        for idx in range(len(reward)):
            player_obs = observation[idx]
            # Check if this player performed a clearance: ball goes from near own goal to farther up the field
            # Assume 'ball_direction' represents the ball movement vector, and 'ball_owned_player' the player index
            # A simple model: if the player kicked the ball out from the defensive third
            if player_obs['ball_owned_player'] == player_obs['active']:
                ball_pos_x = player_obs['ball'][0]
                ball_dir_x = player_obs['ball_direction'][0]

                # Clearance is when the ball is moved significantly towards the opponent's side from the defensive third
                if abs(ball_pos_x) < 0.33 and ball_dir_x > 0.1:
                    additional_rewards["clearance_reward"][idx] += 0.5  # Arbitrary reward for clearances

                # Reward for tackles: assume a ball_owner change without a team change indicates a tackle
                game_mode_reset = player_obs['game_mode'] in [2, 3, 4, 5, 6]  # Resets that might indicate ball contestation
                if game_mode_reset and player_obs['ball_owned_team'] == 0:  # Simplified assumption: our team is always 0
                    additional_rewards["clearance_reward"][idx] += 0.2  # Arbitrary reward for successful tackles

            reward[idx] = reward[idx] + additional_rewards["clearance_reward"][idx]

        return reward, additional_rewards

    def step(self, action):
        """Advance the environment by one step and augment reward information."""
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        # Include custom reward components in the info dict
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
