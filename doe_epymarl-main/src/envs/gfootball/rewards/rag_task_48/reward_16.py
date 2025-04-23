import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that incentivizes optimal high passes from midfield to create direct scoring opportunities.
    Rewards are given based on the player's position (midfield), controlling the ball,
    executing a high pass, and the subsequent positioning and control by other players that could lead to a goal.
    """
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 0.5
        self.scoring_opportunity_reward = 1.0

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['sticky_actions_counter'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward),
                      "scoring_opportunity_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Check if the agent is in midfield
            midfield_threshold = [-0.25, 0.25]  # Defining midfield along the x-axis
            if not midfield_threshold[0] < o['ball'][0] < midfield_threshold[1]:
                continue

            # Check if the agent has control of the ball
            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Check for a high pass
                if o['sticky_actions'][9]:  # assuming index '9' corresponds to high pass action
                    components["high_pass_reward"][rew_index] = self.high_pass_reward
                    reward[rew_index] += self.high_pass_reward
        
                    # Additionally reward if the pass creates a direct scoring opportunity
                    # This can be approximated by checking if another player close to the 
                    # opponent's goal receives the ball immediately after in the observation
                    pass_receiver_pos = observation[(rew_index + 1) % len(reward)]['left_team'][o['designated']]
                    if 0.75 < pass_receiver_pos[0] < 1.0:  # Approximate 'scoring zone'
                        components["scoring_opportunity_reward"][rew_index] = self.scoring_opportunity_reward
                        reward[rew_index] += self.scoring_opportunity_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        # Reset sticky actions counter and update per latest actions seen
        self.sticky_actions_counter.fill(0)
        for i in range(len(obs['sticky_actions'])):
            self.sticky_actions_counter[i] = obs['sticky_actions'][i]
        return observation, reward, done, info
