import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for executing well-timed defensive sliding tackles."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {'sticky_actions_counter': self.sticky_actions_counter.tolist()}
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper']['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            # Check if there is a defensive situation:
            # Own team does not have the ball and the ball is close
            if o['ball_owned_team'] == 1 or o['ball_owned_team'] == -1:
                player_pos = o['left_team'][o['active']]
                ball_pos = o['ball'][:2]
                distance_to_ball = np.linalg.norm(player_pos - ball_pos)

                # Check for executing a tackle (action index for sliding is typically 4)
                if o['sticky_actions'][4] == 1 and distance_to_ball < 0.05:
                    # Reward the tackle based on its closeness to the ball at the moment of the tackle
                    components["tackle_reward"][rew_index] = 0.5 / distance_to_ball
                
                reward[rew_index] += components["tackle_reward"][rew_index]

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
                if action == 1:
                    self.sticky_actions_counter[i] += 1

        return observation, reward, done, info
