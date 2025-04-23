import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that enhances defensive coordination and efficient transitions from defense to attack."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initializing variables for counting how often defense transitions effectively
        self.defensive_transition_counter = np.zeros(5, dtype=int)
        self.expected_defensive_action = 0
        self.defensive_reward_weight = 0.05
        self.attack_transition_reward_weight = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.defensive_transition_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['CheckpointWrapper'] = {'defensive_transitions': self.defensive_transition_counter.tolist()}
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.defensive_transition_counter = np.array(from_pickle['CheckpointWrapper']['defensive_transitions'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_coordination_reward": [0.0] * len(reward),
                      "attack_transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index, o in enumerate(observation):
            # Defensive coordination reward logic
            if o['ball_owned_team'] == observation[rew_index]['left_team_roles'][0] and \
                    o['ball_owned_player'] != 1:  # We check ball possession not with the goal keeper
                if o['game_mode'] in (3, 4, 5):  # We consider free kick, corner and throw in are defensive scenarios
                    self.defensive_transition_counter[rew_index] += 1
                    components['defensive_coordination_reward'][rew_index] = self.defensive_reward_weight

            # Efficient ball distribution reward transitioning to attack
            if self.defensive_transition_counter[rew_index] and o['ball_owned_team'] == 0:
                # The team transitioned from defense to controlling the ball on their side
                components['attack_transition_reward'][rew_index] = self.attack_transition_reward_weight
                self.defensive_transition_counter[rew_index] = 0  # Resetting after successful transition

            # Aggregate rewards
            reward[rew_index] += (components['defensive_coordination_reward'][rew_index] +
                                  components['attack_transition_reward'][rew_index])

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
