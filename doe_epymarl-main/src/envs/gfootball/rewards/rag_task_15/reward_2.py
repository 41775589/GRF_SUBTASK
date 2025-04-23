import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for mastering long passes with precision in the Google Research Football environment."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.max_pass_length = 0.5  # The threshold to consider a pass as a long pass
        self.accuracy_threshold = 0.1  # Distance from a teammate to be considered as an accurate pass

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), 
                      "precision_long_pass_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if o['ball_owned_team'] == 1:  # if the right team owns the ball
                # Get the position and direction of the ball
                ball_pos = o['ball'][:2]
                ball_dir = o['ball_direction'][:2]

                # Calculate the distance travelled by the ball in the X direction (longitudinally)
                ball_travel_distance = np.linalg.norm(ball_dir)
                if ball_travel_distance > self.max_pass_length:
                    # Identify if the pass could be heading towards a teammate
                    for player_pos in o['right_team']:
                        if np.linalg.norm(ball_pos + ball_dir - player_pos) < self.accuracy_threshold:
                            # Reward for accurate long pass
                            components["precision_long_pass_reward"][rew_index] += 0.5
                            reward[rew_index] += components["precision_long_pass_reward"][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Adding reward components to the info dictionary
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        
        # Update sticky actions counter
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        
        return observation, reward, done, info
