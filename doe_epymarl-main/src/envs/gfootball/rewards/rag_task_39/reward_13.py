import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for successful clearances under pressure."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearance_successful = False

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.clearance_successful = False
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['clearance_successful'] = self.clearance_successful
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.clearance_successful = from_pickle['clearance_successful']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # We focus on clearances: Bonus for moving the ball from defensive area during pressure
            opponent_close = any(np.linalg.norm(o['right_team'][i] - o['ball'][:2]) < 0.1
                                 for i in range(len(o['right_team'])))
            own_player_close = any(np.linalg.norm(o['left_team'][i] - o['ball'][:2]) < 0.1
                                   for i in range(len(o['left_team'])))
            in_defensive_third = o['ball'][0] < -0.3
            
            # Providing reward if the clearance is under pressure and successful
            if in_defensive_third and opponent_close and not own_player_close:
                if o['ball_direction'][0] > 0:  # Ball is moving forward
                    components['clearance_reward'][rew_index] = 1.0
                    self.clearance_successful = True
                else:
                    components['clearance_reward'][rew_index] = -0.5
                    self.clearance_successful = False

            reward[rew_index] = components['base_score_reward'][rew_index] + components['clearance_reward'][rew_index]

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
