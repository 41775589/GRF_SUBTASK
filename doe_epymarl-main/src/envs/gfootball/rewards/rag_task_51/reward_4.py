import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that modifies rewards for specialized goalkeeper training.
    The rewards are designed to:
    - Encourage the goalkeeper to stop shots (shot-stopping).
    - Encourage quick reflex actions.
    - Reward initiating counter-attacks with accurate passes from the goalkeeper.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Defining reward adjustments
        self.reflex_bonus = 0.1
        self.save_bonus = 0.3
        self.pass_bonus = 0.2
        self.ball_control_penalty = 0.05

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "reflex_bonus": [0.0] * len(reward),
                      "save_bonus": [0.0] * len(reward),
                      "pass_bonus": [0.0] * len(reward),
                      "ball_control_penalty": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the active player is the goalkeeper
            if o['active'] == 0:  # Typically goalkeeper has the index 0
                if o['ball_owned_team'] == o['active'] and o['game_mode'] == 0: # Normal gameplay
                    # Reward based on sticky actions that imply reflexes
                    if 'sticky_actions' in o and np.sum(o['sticky_actions'][5:]) > 0:  # Reflex actions are indices 5-9
                        components['reflex_bonus'][rew_index] = self.reflex_bonus
                        reward[rew_index] += self.reflex_bonus

                    # Encourage the goalkeeper to control the ball - reward on catch/stop without losing ball
                    if o['ball_owned_player'] == o['active'] and o['ball_direction'][2] < 0:  # Negative z implies ball is coming towards the goalie
                        components['save_bonus'][rew_index] = self.save_bonus
                        reward[rew_index] += self.save_bonus

                    # Encourage passing to initiate counter-attacks
                    if o['ball_owned_player'] == o['active'] and o['right_team_direction'][o['ball_owned_player']][0] > 0.05:  # Positive x-direction movement
                        components['pass_bonus'][rew_index] = self.pass_bonus
                        reward[rew_index] += self.pass_bonus

                # Penalize losing the ball control
                if o['ball_owned_team'] != o['active']:
                    components['ball_control_penalty'][rew_index] = -self.ball_control_penalty
                    reward[rew_index] += -self.ball_control_penalty

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
