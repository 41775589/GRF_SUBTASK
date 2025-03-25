import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a defensive and counter-attack focused dense reward to train agents."""

    def __init__(self, env):
        super().__init__(env)
        self._ball_owner_team_previous_step = None
        self._previous_player_position = None
        self._counter_attack_opportunity = 0
        self._transition_speed = 0.1
        self._steal_bonus = 1.0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self._ball_owner_team_previous_step = None
        self._previous_player_position = None
        self._counter_attack_opportunity = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['counter_attack_opportunity'] = self._counter_attack_opportunity
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self._counter_attack_opportunity = from_pickle['counter_attack_opportunity']
        return from_pickle

    def reward(self, reward):
        """Augments the reward based on defensive awareness and counter-attack potential."""

        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "defensive_reward": [0.0] * len(reward),
            "counter_attack_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Defensive play: reward intercepting the ball
            if o['ball_owned_team'] != self._ball_owner_team_previous_step and o['ball_owned_team'] == 0:
                if self._ball_owner_team_previous_step == 1:
                    components["defensive_reward"][rew_index] += self._steal_bonus
                    reward[rew_index] += components["defensive_reward"][rew_index]

            # Counter-attack opportunity: evaluate transition from defense to attack
            if self._previous_player_position is not None:
                player_pos = o['left_team'][o['active']]
                player_prev_pos = self._previous_player_position
                distance_moved = np.linalg.norm(player_pos - player_prev_pos)
                transition_speed = distance_moved / o['steps_left']
                self._counter_attack_opportunity += transition_speed * self._transition_speed

                components["counter_attack_reward"][rew_index] = self._counter_attack_opportunity
                reward[rew_index] += components["counter_attack_reward"][rew_index]

            # Update tracking variables
            self._ball_owner_team_previous_step = o['ball_owned_team']
            self._previous_player_position = o['left_team'][o['active']]

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
