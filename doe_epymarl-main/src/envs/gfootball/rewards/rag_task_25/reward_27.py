import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specific reward based on dribbling and sprint actions, promoting ball control and evasion under pressure."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "dribbling_bonus": [0.0] * len(reward),
                      "sprint_bonus": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage dribbling when in possession of the ball.
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                dribbling_active = o['sticky_actions'][9]  # 'action_dribble' index is 9
                if dribbling_active:
                    components["dribbling_bonus"][rew_index] = 0.05  # Increase reward for dribbling
                    reward[rew_index] += components["dribbling_bonus"][rew_index]

            # Encouraging the use of sprints effectively.
            sprint_active = o['sticky_actions'][8]  # 'action_sprint' index is 8
            if sprint_active:
                components["sprint_bonus"][rew_index] = 0.02  # Sprint bonus
                reward[rew_index] += components["sprint_bonus"][rew_index]
                
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
