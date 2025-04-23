import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on offensive play between midfielders and strikers."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.pass_completion_reward = 0.5
        self.goal_completion_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {'base_score_reward': reward.copy(),
                      'pass_completion_reward': [0.0] * len(reward),
                      'goal_completion_reward': [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            midfielders = [i for i, role in enumerate(o['right_team_roles']) if role in (4, 5, 6)]
            strikers = [i for i, role in enumerate(o['right_team_roles']) if role == 9]

            if o['ball_owned_team'] == 1:  # If the right team owns the ball
                if o['ball_owned_player'] in midfielders:
                    # Check if the pass is completed to a striker
                    if any([o['sticky_actions'][9] == 1 for i in strikers]):  # Assuming 9 indicates a pass action
                        components['pass_completion_reward'][rew_index] = self.pass_completion_reward
                        reward[rew_index] += self.pass_completion_reward

                elif o['ball_owned_player'] in strikers and o['score'][0] != o['score'][1]:
                    components['goal_completion_reward'][rew_index] = self.goal_completion_reward
                    reward[rew_index] += self.goal_completion_reward

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
