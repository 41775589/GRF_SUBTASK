import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the reward signal for defensive coordination tasks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_switch_counter = {}
        self.previous_ball_owner = None
        self.scoring_opportunities_reward = 0.2
        self.transition_reward = 0.1

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.possession_switch_counter = {}
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.possession_switch_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.possession_switch_counter = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for i, rew in enumerate(reward):
            o = observation[i]

            new_owner_team = o.get('ball_owned_team')
            
            # Reward for gaining possession from a defending scenario
            if new_owner_team != self.previous_ball_owner and new_owner_team == 0:  # if team 0 now has possession
                if self.previous_ball_owner == 1:  # and previously it was owned by team 1
                    reward[i] += self.transition_reward
                    components["defensive_transition_reward"][i] += self.transition_reward

            # Identify strategic plays that prevent scoring
            if o.get('game_mode') in [3, 4, 5, 6]:  # FreeKick, Corner, ThrowIn, Penalty by the opponent
                reward[i] += self.scoring_opportunities_reward
                components["defensive_transition_reward"][i] += self.scoring_opportunities_reward

            self.previous_ball_owner = new_owner_team

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
                self.sticky_actions_counter[i] = int(action)
        return observation, reward, done, info
