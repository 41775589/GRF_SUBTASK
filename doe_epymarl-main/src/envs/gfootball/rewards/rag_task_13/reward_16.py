import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that focuses on defensive skills such as intense man-marking, blocking shots,
       and stalling forward moves of opposing players, commonly known as the 'stopper' role."""

    def __init__(self, env):
        super().__init__(env)
        # Tracking player efforts of intercepting and blocking.
        self.interrupts_counter = 0
        self.block_counter = 0
        self.defensive_reward = 0.1  # Reward given for effective defensive actions
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.interrupts_counter = 0
        self.block_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'interruptions': self.interrupts_counter,
                                                'blocks': self.block_counter}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.interrupts_counter = from_pickle['CheckpointRewardWrapper']['interruptions']
        self.block_counter = from_pickle['CheckpointRewardWrapper']['blocks']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defense_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Encourage interrupting opponent's ball possession
            if o['ball_owned_team'] == 1:  # Opponent has the ball
                opponent_ball_handler = (o['ball_owned_player'] ==
                                         o['right_team_roles'].index(9))  # Check if forward
                if opponent_ball_handler:
                    self.interrupts_counter += 1
                    reward[rew_index] += self.defensive_reward
                    components["defense_reward"][rew_index] = self.defensive_reward

            # Reward blocking opponent's shot
            if o['game_mode'] in [3, 4]:  # free kick or corner, high chance of opponent shot
                self.block_counter += 1
                reward[rew_index] += self.defensive_reward
                components["defense_reward"][rew_index] = self.defensive_reward

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
