import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that provides a reward for successfully completing high passes from midfield that lead to scoring opportunities.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.high_pass_acc = 0.2  # Adjustable coefficient for high pass accuracy reward.

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle.get('sticky_actions_counter', np.zeros(10, dtype=int))
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(), "high_pass_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            # If the team performed a high pass from midfield to a near-scoring position.
            if ('ball_owned_team' in o and o['ball_owned_team'] == 0 and
                'active' in o and
                'sticky_actions' in o and
                o['sticky_actions'][9] == 1 and  # checking if the dribble action was active
                (-0.2 <= o['ball'][0] <= 0.2) and (0.1 <= abs(o['ball'][1]) < 0.4)):  # ball in midfield range but towards goals
                towards_goal_x_coord = 1 if o['ball'][0] > 0 else -1
                
                # Check if the target of high pass is closer to goal line.
                if any(player[0] * towards_goal_x_coord > o['ball'][0] * towards_goal_x_coord and 
                       0.85 < abs(player[0]) < 1 for player in o['right_team']):
                    additional_reward = self.high_pass_acc
                    reward[rew_index] += additional_reward
                    components["high_pass_reward"][rew_index] += additional_reward

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        
        # Track sticky actions.
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]
        
        return observation, reward, done, info
