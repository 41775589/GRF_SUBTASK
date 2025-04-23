import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that enhances goalkeeper specific training in the Google Research Football Environment.
    Focus areas include:
    - Shot-stopping
    - Quick reflexes
    - Initiating counter-attacks with accurate passes
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.goalkeeper_index = 0  # Assuming that the goalkeeper is always the first player in observation for simplicity

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()
    
    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        return state

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        """
        Modifies the reward focused on the tasks for a goalkeeper.
        1. Bonus for saving goals.
        2. Penalty for conceding goals.
        3. Bonus for successful clearances/pass initiating counter-attacks.
        """
        observation = self.env.unwrapped.observation()
        # Initialize component bonuses
        components = {"base_score_reward": reward.copy(), "save_bonus": [0.0]*len(reward), "clearance_bonus": [0.0]*len(reward)}

        if observation is None:
            return reward, components

        for rew_index, (o, rew) in enumerate(zip(observation, reward)):
            # Set specific rewards/punishments for the goalkeeper assuming goalkeeper observation is at index 0
            if rew_index == self.goalkeeper_index:
                components["save_bonus"][rew_index] = 0
                components["clearance_bonus"][rew_index] = 0
                
                # Detect goal-saving - heuristically based on the ball's proximity and velocity towards own goal
                if o['ball_owned_team'] == 1 and o['ball'][0] > 0.8:  # Ball in goalkeeper's half near goal
                    if np.linalg.norm(o['ball_direction']) > 0:  # Ball is moving
                        components["save_bonus"][rew_index] = 0.5  # goalkeeper saves the ball, reward him
                
                # Adding clearance reward
                if o['ball_owned_team'] == 0 and o['ball_owned_player'] == 0:  # ball owned by goalie
                    if any(o['sticky_actions'][0:4]):  # goalie performs a clearance/pass action
                        components["clearance_bonus"][rew_index] = 0.3  # Successful clearance initiated

                # Adjusting reward for the goalkeeper's action
                reward[rew_index] += (components["save_bonus"][rew_index] + components["clearance_bonus"][rew_index])

        return reward, components

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        # Pass the raw reward to the modifier
        reward, components = self.reward(reward)
        # Append the modified reward and its components back into info for tracking and debugging
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        return obs, reward, done, info
