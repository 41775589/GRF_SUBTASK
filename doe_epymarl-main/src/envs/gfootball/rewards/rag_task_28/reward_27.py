import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards dribbling skills in face-to-face situations with the goalkeeper,
    emphasizing quick feints, sudden direction changes, and maintaining ball control under pressure.
    """

    def __init__(self, env):
        super().__init__(env)
        self.last_ball_position = None
        self.dribble_reward_factor = 1.0
        self.pressure_reward_factor = 0.5
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.last_ball_position = None
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['last_ball_position'] = self.last_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_position = from_pickle.get('last_ball_position', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "dribble_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index in range(len(reward)):
            current_observation = observation[index]
            # Dribble and pressure successful handling.
            if current_observation['ball_owned_team'] == 0 and current_observation['active'] == current_observation['ball_owned_player']:
                ball_position = current_observation['ball']
                if self.last_ball_position is not None:
                    # Calculate distance moved with the ball
                    dribble_distance = np.linalg.norm(np.array(ball_position[:2]) - np.array(self.last_ball_position[:2]))
                    components["dribble_reward"][index] = dribble_distance * self.dribble_reward_factor
                reward[index] += components["dribble_reward"][index]

                # Reward handling pressure: being close to the goalkeeper (i.e., opponent's goal area)
                if abs(ball_position[0]) > 0.8:  # Goalkeeper's area is assumed to be close to x = 1 or x = -1
                    reward[index] += self.pressure_reward_factor

            self.last_ball_position = current_observation['ball']

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
