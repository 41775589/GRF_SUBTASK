import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that rewards agents for executing high passes effectively from midfield to create scoring chances.
    """
    
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.goal_regions = np.linspace(-0.42, 0.42, 5)  # Dividing the goal width into regions
        self.high_pass_reward = 0.3
        self.scoring_chance_reward = 0.5
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
                      "high_pass_reward": [0.0, 0.0],
                      "scoring_chance_reward": [0.0, 0.0]}
        
        if observation is None:
            return reward, components

        for agent_index, o in enumerate(observation):
            if o['game_mode'] in [2, 3, 4, 6]:  # Handling non-play modes like free kicks, corners
                continue

            if o['ball_owned_team'] == 0 and o['ball_owned_player'] == o['active']:
                # Agent has ball control
                if o['ball'][2] > 0.15:  # Consider it a 'high' pass if ball z > 0.15
                    components["high_pass_reward"][agent_index] = self.high_pass_reward
                    reward[agent_index] += components["high_pass_reward"][agent_index]

                # Check if the pass leads to a potential scoring opportunity
                if o['right_team'][o['designated']][0] > 0.5:  # Player in opponent's half
                    y_pos = o['right_team'][o['designated']][1]
                    for region in self.goal_regions:
                        if abs(y_pos - region) < 0.1:  # Check if player is within a goal region
                            components["scoring_chance_reward"][agent_index] = self.scoring_chance_reward
                            reward[agent_index] += components["scoring_chance_reward"][agent_index]
                            break

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
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        return observation, reward, done, info
