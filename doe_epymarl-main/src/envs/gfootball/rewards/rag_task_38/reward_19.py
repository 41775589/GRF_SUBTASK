import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for initiating counterattacks post-defense through accurate long passes and quick transitions."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_accuracy_threshold = 0.6
        self.transition_speed_threshold = 0.02

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "counterattack_bonus": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for i in range(len(reward)):
            o = observation[i]
            
            # Elements needed from observation to calculate the reward
            ball_pos = o['ball'][0]  # x coordinate of the ball
            ball_direction = o['ball_direction'][0]  # x direction of the ball
            ball_speed = np.linalg.norm(o['ball_direction'])  # speed of the ball
            ball_owned_team = o['ball_owned_team']
            
            # Encourage long forward passes and fast transitions from defense to attack
            if ball_owned_team == 0 and ball_pos < 0:  # ball in the left team's possession and in their half
                if ball_speed > self.transition_speed_threshold and ball_direction > 0:
                    # Ball is moving quickly towards the attacking half
                    components["counterattack_bonus"][i] = 0.5
                
                if ball_pos > self.pass_accuracy_threshold and ball_speed > self.transition_speed_threshold:
                    # Reward accurate long passes that reach the attacking half
                    components["counterattack_bonus"][i] += 0.5
            
            reward[i] += components["counterattack_bonus"][i] * reward[i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Collect information about reward components for further analysis
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        # Collect sticky actions to monitor player behavior further
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action

        return observation, reward, done, info
