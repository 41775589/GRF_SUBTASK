import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward focused on fostering counterattacks 
    by rewarding precise long passes and rapid transitions from defense to attack.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position_x = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_position_x = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position_x'] = self.last_ball_position_x
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position_x = from_pickle.get('last_ball_position_x', None)
        return from_pickle

    def reward(self, reward):
        """
        Calculate a new reward considering:
        - Successful long passes
        - Quick position changes in ball ownership from defense to attack
        """
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "long_pass_reward": [0.0] * len(reward),
                      "counterattack_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_x, ball_y, _ = o['ball']

            # Previous position needed to evaluate the direction of the ball
            if self.last_ball_position_x is not None:
                moved_forward = (ball_x - self.last_ball_position_x) * (1 if o['ball_owned_team'] == 0 else -1)

                # Check ball moving radically forwards after a long pass
                if moved_forward > 0.3:
                    components["long_pass_reward"][rew_index] = 0.2
                    reward[rew_index] += components["long_pass_reward"][rew_index]

            # Counterattack bonus: rapid transition from own half to the opponent's half
            if o['ball_owned_team'] == 0 and ball_x > 0.5 and self.last_ball_position_x < -0.5:
                components["counterattack_bonus"][rew_index] = 0.5
                reward[rew_index] += components["counterattack_bonus"][rew_index]

            self.last_ball_position_x = ball_x

        return reward, components

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
