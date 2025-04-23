import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward based on cooperative plays between shooters and passers, 
    encouraging efficient ball passing and shooting collaboration, leading to scoring opportunities.
    """

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # Counters for trackers of specific in-game events.
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Reward increments for collaboration and scoring.
        self.passing_reward = 0.05
        self.scoring_opportunity_reward = 0.2
        self.last_ball_owner = None

    def reset(self):
        self.sticky_actions_counter.fill(0)
        self.last_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.last_ball_owner
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_owner = from_pickle['CheckpointRewardWrapper']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_reward": [0.0] * len(reward),
                      "scoring_opportunity_reward": [0.0] * len(reward)}
                      
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            cur_obs = observation[rew_index]

            # Check if a goal was scored
            if cur_obs['score'][1] > cur_obs['score'][0]:  # Assuming index 1 is the controlled team
                if self.last_ball_owner is not None and cur_obs['active'] == self.last_ball_owner:
                    # Reward the scorer
                    components["scoring_opportunity_reward"][rew_index] += self.scoring_opportunity_reward
                    reward[rew_index] += components["scoring_opportunity_reward"][rew_index]

            if cur_obs['ball_owned_team'] == 1 and cur_obs['ball_owned_player'] == cur_obs['active']:
                # If current possession by the agent team and the player is active
                if self.last_ball_owner is not None and self.last_ball_owner != cur_obs['active']:
                    # Reward for a change in possession which indicates a pass
                    components["passing_reward"][rew_index] += self.passing_reward
                    reward[rew_index] += components["passing_reward"][rew_index]
                # Update the last ball owner
                self.last_ball_owner = cur_obs['active']

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
