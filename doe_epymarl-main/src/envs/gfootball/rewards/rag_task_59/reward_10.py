import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that improves goalkeeper training with back-up and efficient clear-ball coordination."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['checkpoint_data'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['checkpoint_data']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goalkeeper_efficiency": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            if 'ball_owned_team' in o and o['ball_owned_team'] == 0:
                # Assuming goalkeeper is always the first player in the left team list
                goalkeeper_pos = o['left_team'][0]
                ball_pos = o['ball'][:2]

                dist_to_ball = np.linalg.norm(goalkeeper_pos - ball_pos)
                is_goalkeeper = (o['ball_owned_player'] == 0)

                if is_goalkeeper and dist_to_ball <= 0.1:
                    components["goalkeeper_efficiency"][rew_index] = 0.5
                    reward[rew_index] += components["goalkeeper_efficiency"][rew_index]

                # Bonus for clear distribution if goalkeeper has the ball and clears it well
                if 'ball_owned_player' in o and o['ball_owned_player'] == 0:
                    if 'action' in o and o['action'] == 'long_pass':
                        components["goalkeeper_efficiency"][rew_index] = 1.0
                        reward[rew_index] += components["goalkeeper_efficiency"][rew_index]

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
