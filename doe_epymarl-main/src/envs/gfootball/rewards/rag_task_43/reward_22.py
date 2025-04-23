import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper to add rewards for defensive positioning and quick counterattacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)  # Track for sticky actions used by each player.
        self.defensive_positions_collected = {}
        self.counterattack_positions_collected = {}
        self.defensive_reward = 0.1
        self.counterattack_reward = 0.2

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.defensive_positions_collected = {}
        self.counterattack_positions_collected = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['defensive_positions'] = self.defensive_positions_collected
        to_pickle['counterattack_positions'] = self.counterattack_positions_collected
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_positions_collected = from_pickle['defensive_positions']
        self.counterattack_positions_collected = from_pickle['counterattack_positions']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward),
            "counterattack_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for index, o in enumerate(observation):
            # Increase defensive reward if the player is in a good defensive position.
            if o['left_team_active'][o['active']] and o['ball_owned_team'] == 0:  # player from left team and ball owned by left team
                player_pos = o['left_team'][o['active']]
                if player_pos[0] < -0.7:  # defensive position toward own goal
                    if index not in self.defensive_positions_collected:
                        components["defensive_reward"][index] = self.defensive_reward
                        reward[index] += components["defensive_reward"][index]
                        self.defensive_positions_collected[index] = True

            # Increase counterattack reward by positioning for fast break if loss of ball.
            if o['left_team_active'][o['active']] and o['ball_owned_team'] == 1:  # player from left team and ball owned by the opponent
                player_pos = o['left_team'][o['active']]
                if player_pos[0] > 0:  # advanced field position
                    if index not in self.counterattack_positions_collected:
                        components["counterattack_reward"][index] = self.counterattack_reward
                        reward[index] += components["counterattack_reward"][index]
                        self.counterattack_positions_collected[index] = True

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
