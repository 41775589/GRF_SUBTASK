import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward component based on effective ball distribution during defensive play
    under pressure, emphasizing tactical short passes to maintain ball possession and facilitate counter-attacks.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter.fill(0)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "possession_reward": [0.0] * len(reward),
                      "pass_precision_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        # Assume reward and observation have length of the agents (2 in this case)
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            score_before = o['score']
            self.sticky_actions_counter += o['sticky_actions']

            # Assessing effective ball possession and distribution under pressure
            if o['ball_owned_team'] == rew_index:  # if the ball is owned by the team of this agent
                components['possession_reward'][rew_index] = 0.01  # Reward for maintaining possession

                # Encourage strategic short passing by checking if a pass action is active and successful
                if 'action_short_pass' in o['sticky_actions'] and o['sticky_actions']['action_short_pass']:
                    if o['ball_owned_player'] != o['active']:  # Ball successfully passed to another player
                        components['pass_precision_reward'][rew_index] = 0.05  # Reward for a successful pass

            # Update composite reward for the specific agent
            reward[rew_index] = (
                reward[rew_index] +
                components['possession_reward'][rew_index] +
                components['pass_precision_reward'][rew_index]
            )

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        self.sticky_actions_counter.fill(0)
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
