import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for executing high passes from midfield that create direct scoring opportunities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_reward = 1.0
        self.pass_effectiveness_threshold = 0.5  # Hypothetical threshold to consider a pass 'effective'

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['high_pass_action_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['high_pass_action_counter']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.get_obs()
        components = {"base_score_reward": reward.copy(),
                      "high_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for agent_index in range(len(reward)):
            o = observation[agent_index]
            
            if o['game_mode'] in [4, 6]:  # Considering high-pass scenarios during corners and penalties.
                player_position = o['left_team'][o['active']]
                ball_position = o['ball'][:2]

                # Check if the player is in the midfield region
                if -0.3 <= player_position[0] <= 0.3:
                    pass_distance = np.linalg.norm(player_position - ball_position)
                    # Reward high and long passes from midfield if they lead directly or indirectly to goal opportunities
                    if pass_distance > self.pass_effectiveness_threshold:
                        components["high_pass_reward"][agent_index] = self.high_pass_reward
                        reward[agent_index] += components["high_pass_reward"][agent_index]
        
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
