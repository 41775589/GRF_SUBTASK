import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Number of checkpoints for catching the ball and positioning during defensive tackles
        self._num_checkpoints = 5
        self._checkpoint_reward = 0.2
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_success = {}
    
    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._tackle_success = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self._tackle_success
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._tackle_success = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        components = {"base_score_reward": reward.copy(), 
                      "tackle_reward": [0.0] * len(reward)}
        
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Scenario specific observations like position and game mode check
            if 'left_team' in o and 'game_mode' in o and 'ball_owned_team' in o:
                # Encouraging tackles in defensive zones
                if o['ball_owned_team'] == 1 and o['game_mode'] == 0 and not self._tackle_success.get(rew_index, False):
                    player_position = o['left_team'][o['active']]
                    own_goal_x = -1  # X-coordinate of our team's goal
                    ball_position = o['ball'][:2]

                    # Calculate distances to our own goal
                    distance_to_goal = np.float64(np.linalg.norm(player_position - [own_goal_x, 0]))
                    ball_distance_to_goal = np.float64(np.linalg.norm(ball_position - [own_goal_x, 0]))

                    # Check if the player is in the defensive third and close to the ball
                    if distance_to_goal < 1/3 and np.abs(distance_to_goal - ball_distance_to_goal) < 0.1:
                        components["tackle_reward"][rew_index] = self._checkpoint_reward
                        self._tackle_success[rew_index] = True  
                        
            reward[rew_index] += components["tackle_reward"][rew_index]
        
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        return observation, reward, done, info
