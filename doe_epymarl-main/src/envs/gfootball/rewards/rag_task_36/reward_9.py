import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized reward function for dribbling maneuvers and dynamic 
    positioning to facilitate transitions between defense and offense."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_stat = [0, 0]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_stat = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['dribble_stat'] = self.dribble_stat
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribble_stat = from_pickle.get('dribble_stat', [0, 0])
        return from_pickle

    def reward(self, reward):
        """Augment rewards based on dribbling skills and dynamic positioning."""
        observation = self.env.unwrapped.observation()
        if observation is None:
            return reward, {}

        components = {"base_score_reward": reward.copy(),
                      "dribble_reward": [0.0] * len(reward),
                      "positioning_reward": [0.0] * len(reward)}
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            dribbling_action = o['sticky_actions'][9]  # Assuming index 9 is the dribble action.

            # Reward for starting and stopping dribbling effectively.
            if dribbling_action and not self.dribble_stat[rew_index]:
                components['dribble_reward'][rew_index] = 0.1  # reward starting dribble
                self.dribble_stat[rew_index] = 1
            elif not dribbling_action and self.dribble_stat[rew_index]:
                components['dribble_reward'][rew_index] = 0.1  # reward stopping dribble
                self.dribble_stat[rew_index] = 0

            # Dynamic positioning reward: encourage players to move towards the ball
            player_pos = o['left_team'][o['active']] if o['active'] < o['right_team'].shape[0] else o['right_team'][o['active']]
            ball_pos = o['ball'][:2]
            distance_to_ball = np.linalg.norm(player_pos - ball_pos)
            
            components['positioning_reward'][rew_index] = (0.5 / (distance_to_ball + 0.1))  # reward getting closer to the ball

            # Combine rewards
            reward[rew_index] += components['dribble_reward'][rew_index]
            reward[rew_index] += components['positioning_reward'][rew_index]

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
