import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for sprinting and enhancing defensive coverage."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._num_defensive_zones = 5
        self._defensive_rewards = np.linspace(0.1, 0.5, self._num_defensive_zones)
        self._defensive_zones_collected = [set() for _ in range(2)]
        self._prev_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._defensive_zones_collected = [set() for _ in range(2)]
        self._prev_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_zones'] = self._defensive_zones_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._defensive_zones_collected = from_pickle['defensive_zones']
        return from_pickle

    def reward(self, rewards):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": rewards.copy(),
                      "defensive_zone_reward": [0.0] * len(rewards)}
        if observation is None:
            return rewards, components
        
        assert len(rewards) == len(observation)

        for index, (o, reward) in enumerate(zip(observation, rewards)):
            ball_position = o['ball'][:2]  # get x, y position of the ball
            if self._prev_ball_position is not None:
                movement_vector = np.array(ball_position) - np.array(self._prev_ball_position)
                if np.linalg.norm(movement_vector) > 0:  # Check if the ball has moved
                    for i in range(self._num_defensive_zones):
                        # Calculating defensive lines based on y-coordinate to divide the field vertically
                        defensive_line = -1 + i * 0.4 / self._num_defensive_zones
                        if abs(ball_position[1]) < defensive_line and i not in self._defensive_zones_collected[index]:
                            components["defensive_zone_reward"][index] += self._defensive_rewards[i]
                            self._defensive_zones_collected[index].add(i)
            
            self._prev_ball_position = ball_position

            # Incorporate sprint action encouragement
            if 'sticky_actions' in o and o['sticky_actions'][8]:  # sprint is index 8
                rewards[index] += 0.02  # small constant reward for using sprint
            # Update the reward with calculated component rewards
            rewards[index] += components["defensive_zone_reward"][index]

        return rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_flag in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_flag
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
