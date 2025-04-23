import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A reward wrapper that focuses on defensive tackles."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.foul_penalty = -0.5
        self.tackle_reward = 0.3
        self.prev_ball_owned_team = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_owned_team = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'prev_ball_owned_team': self.prev_ball_owned_team
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions_counter']
        self.prev_ball_owned_team = from_pickle['prev_ball_owned_team']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0.0] * len(reward),
                      "foul_penalty": [0.0] * len(reward)}

        for rew_index, _ in enumerate(reward):
            o = observation[rew_index]

            # Applying tackle rewards and penalties
            if o['game_mode'] == 3:  # FreeKick mode
                if self.prev_ball_owned_team == 0 and o['ball_owned_team'] == 1:
                    # foul by left team
                    components['foul_penalty'][rew_index] = self.foul_penalty
                elif self.prev_ball_owned_team == 1 and o['ball_owned_team'] == 0:
                    # foul by right team
                    components['foul_penalty'][rew_index] = self.foul_penalty

            # Reward for successful tackle without fouling
            if o['game_mode'] == 0:  # Normal Play
                if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # dribble active
                    components['tackle_reward'][rew_index] = self.tackle_reward

            # Update the reward
            adjusted_reward = (reward[rew_index] +
                               components['tackle_reward'][rew_index] +
                               components['foul_penalty'][rew_index])
            reward[rew_index] = adjusted_reward
        
        # Update ball ownership for next step
        if observation:
            self.prev_ball_owned_team = observation[0]['ball_owned_team']

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
