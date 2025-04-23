import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense checkpoint reward for executing long passes."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.pass_received = [False, False]  # Track if a pass has been completed for each agent
        self.last_ball_holder = None          # Track the last agent that held the ball
        self.pass_reward_coefficient = 1.0    # Coefficient for the reward of completed pass

    def reset(self):
        self.pass_received = [False, False]
        self.last_ball_holder = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['PassRewardWrapper'] = {
            'pass_received': self.pass_received,
            'last_ball_holder': self.last_ball_holder
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        loaded_state = from_pickle['PassRewardWrapper']
        self.pass_received = loaded_state['pass_received']
        self.last_ball_holder = loaded_state['last_ball_holder']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Ensure ball is owned by the active team and there is a different ball holder
            if o['ball_owned_team'] in [0, 1] and o['ball_owned_player'] != -1 and o['ball_owned_player'] != self.last_ball_holder:
                if self.last_ball_holder is not None:
                    # Calculate distance between the last and current ball holder
                    last_ball_pos = self.env.unwrapped.observation()[1-self.last_ball_holder]['ball']
                    current_ball_pos = o['ball']
                    distance = np.linalg.norm(np.array(last_ball_pos[:2]) - np.array(current_ball_pos[:2]))
                    
                    # Check if the pass was long enough
                    if distance > 0.3:
                        self.pass_received[rew_index] = True
                        components['long_pass_reward'][rew_index] = self.pass_reward_coefficient
                        reward[rew_index] += components['long_pass_reward'][rew_index]

                # Update last ball holder
                self.last_ball_holder = rew_index

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
