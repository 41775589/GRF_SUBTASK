import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense checkpoint reward focused on offensive strategies:
    mastering accurate shooting, effective dribbling to evade opponents, and practicing different pass types.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_success_count = 0
        self.shot_accuracy_reward = 0.2
        self.dribble_evade_reward = 0.15
        self.pass_accuracy_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_success_count = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'dribble_success_count': self.dribble_success_count
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.dribble_success_count = from_pickle['CheckpointRewardWrapper']['dribble_success_count']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "shot_accuracy_reward": [0.0] * len(reward),
                      "dribble_evade_reward": [0.0] * len(reward),
                      "pass_break_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_team = o.get('ball_owned_team', -1)

            # Reward for effectively directed shots: Shot toward the goal
            if ball_owned_team == 0 and o.get('game_mode') == 6:  # Penalty kick (only time a direct shot is guaranteed)
                components['shot_accuracy_reward'][rew_index] = self.shot_accuracy_reward
                reward[rew_index] += self.shot_accuracy_reward

            # Reward for dribbling past an opponent (assumed when ball possession changes under pressure)
            if o.get('sticky_actions')[9] == 1:  # Check if dribble action is active
                self.dribble_success_count += 1
                if self.dribble_success_count >= 3:  # Arbitrarily assume a successful evade after 3 dribbles
                    components['dribble_evade_reward'][rew_index] = self.dribble_evade_reward
                    reward[rew_index] += self.dribble_evade_reward
                    self.dribble_success_count = 0  # reset counter

            # Reward for successful long or high passes that effectively change the play dynamic (caught by a teammate)
            if o['game_mode'] == 2 or o['game_mode'] == 3:  # Consider free kicks and corner kicks as opportunity for strategic passes
                components['pass_break_reward'][rew_index] = self.pass_accuracy_reward
                reward[rew_index] += self.pass_accuracy_reward

        return reward, components
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_count = obs['sticky_actions']
        return observation, reward, done, info
