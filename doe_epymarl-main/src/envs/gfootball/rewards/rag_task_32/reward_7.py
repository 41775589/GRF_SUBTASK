import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for wingers crossing and sprinting from the wings accurately."""

    def __init__(self, env):
        super().__init__(env)
        self._checkpoints = 10  # number of sections the sidelines are divided into for crossing
        self.crossing_reward = 0.2  # reward for each successful progression in crossing
        self.sprint_reward = 0.1  # reward for sprinting along the sideline
        self._last_crossed_checkpoint = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self._last_crossed_checkpoint = {}
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        """Custom reward function focused on wing play."""
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "crossing_reward": [0.0] * len(reward),
                      "sprinting_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, o in enumerate(observation):
            if o['active'] == -1 or o['designated'] == -1:
                continue  # skip if there's no active or designated player in control

            is_winger = (o['left_team_roles'][o['designated']] in [7, 9])  # Assuming roles 7 & 9 are wingers
            if not is_winger:
                continue
            
            ball_owned_by_designated = (o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['designated'])
            if ball_owned_by_designated:
                position = o['left_team'][o['designated']]
                y_pos = position[1]
                
                # Boundary checks for y-axis to see if near the sideline
                if abs(y_pos) > 0.7:
                    # Calculate current checkpoint based on x position:
                    checkpoint = int(position[0] * 5) + 5  # Converts positions to checkpoint index (0-10)
                    last_checkpoint = self._last_crossed_checkpoint.get(index, -1)
                    
                    if checkpoint > last_checkpoint:
                        # Reward for moving forward along the sideline
                        components["crossing_reward"][index] = (checkpoint - last_checkpoint) * self.crossing_reward
                        self._last_crossed_checkpoint[index] = checkpoint

                    # Check for sprinting action being employed
                    if o['sticky_actions'][8] == 1:  # assuming index 8 is for sprinting
                        components["sprinting_reward"][index] = self.sprint_reward
            
            # Update composite reward
            reward[index] += components["crossing_reward"][index] + components["sprinting_reward"][index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = action
        return observation, reward, done, info
