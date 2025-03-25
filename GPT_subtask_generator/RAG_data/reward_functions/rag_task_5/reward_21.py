import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that incentivizes tactical defensive positioning and quick transition for counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the number of checkpoints and reward for each checkpoint
        self.defensive_zones = 5
        self.counter_attack_zones = 5
        self.zone_reward = 0.05
        self._zones_visited_defense = set()
        self._zones_visited_attack = set()

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._zones_visited_defense = set()
        self._zones_visited_attack = set()
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['checkpoint_defense'] = list(self._zones_visited_defense)
        state['checkpoint_attack'] = list(self._zones_visited_attack)
        return state

    def set_state(self, state):
        from_state = self.env.set_state(state)
        self._zones_visited_defense = set(from_state['checkpoint_defense'])
        self._zones_visited_attack = set(from_state['checkpoint_attack'])
        return from_state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "defensive_reward": [0.0] * len(reward), "counter_attack_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            if 'ball_owned_team' not in o or o['ball_owned_team'] != 0:
                continue

            # Encourage players to maintain good defensive positions
            defensive_index = self.assign_zone_defensive(o['left_team'][o['active']])
            if defensive_index not in self._zones_visited_defense:
                self._zones_visited_defense.add(defensive_index)
                reward[rew_index] += self.zone_reward
                components['defensive_reward'][rew_index] = self.zone_reward

            # Encourage quick transition to counter-attack positions
            attack_index = self.assign_zone_attack(o['left_team'][o['active']])
            if attack_index not in self._zones_visited_attack:
                self._zones_visited_attack.add(attack_index)
                reward[rew_index] += self.zone_reward
                components['counter_attack_reward'][rew_index] = self.zone_reward

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
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info

    def assign_zone_defensive(self, position):
        """ Assign defensive zones based on y-coordinate of the player's position """
        y_coord = position[1]
        if y_coord >= 0.3:
            return 1
        elif y_coord >= 0.1:
            return 2
        elif y_coord >= -0.1:
            return 3
        elif y_coord >= -0.3:
            return 4
        else:
            return 5

    def assign_zone_attack(self, position):
        """ Assign counter-attack zones based on y-coordinate of the player's position """
        y_coord = position[1]
        if y_coord >= 0.3:
            return 1
        elif y_coord >= 0.1:
            return 2
        elif y_coord >= -0.1:
            return 3
        elif y_coord >= -0.3:
            return 4
        else:
            return 5
