import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that rewards agents for effective ball transition and control under pressure."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "skill_transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        # Iterate through each agent's observation
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_control = o['ball_owned_player'] == o['active'] and o['ball_owned_team'] == 0

            if ball_control:
                pass_effectiveness = (o['sticky_actions'][8] +          # action_sprint
                                      o['sticky_actions'][9]) * 0.1     # action_dribble

                # Encourage forward movement while controlling the ball
                for i, r_player in enumerate(o['right_team']):
                    if np.linalg.norm(o['ball'][:2] - r_player[:2]) < 0.3:   # proximity to a defending player
                        pass_effectiveness += 0.2
                        break

                components['skill_transition_reward'][rew_index] = pass_effectiveness

            reward[rew_index] += components['skill_transition_reward'][rew_index]

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
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
