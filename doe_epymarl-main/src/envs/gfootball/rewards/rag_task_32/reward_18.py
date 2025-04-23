import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A reward wrapper that encourages wingers to sprint along the wings, 
    dribble past defenders, and accurately perform crosses into the box.
    """

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.crossing_zones = [0.8, 1]  # y-coordinates defining cross attempt zones on each wing
        self.crossing_reward = 0.5       # Reward for a successful cross within the zones
        self.sprint_reward = 0.1         # Incremental reward for sprinting
        self.dribble_reward = 0.1        # Incremental reward for dribbling past a defender
        
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
                      "sprint_reward": [0.0] * len(reward),
                      "dribble_reward": [0.0] * len(reward),
                      "cross_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward sprinting actions
            if o['sticky_actions'][8]:  # action_sprint is indexed at 8
                components["sprint_reward"][rew_index] = self.sprint_reward
                reward[rew_index] += components["sprint_reward"][rew_index]
            
            # Reward dribbling past opponents
            if o['sticky_actions'][9]:  # action_dribble is indexed at 9
                components["dribble_reward"][rew_index] = self.dribble_reward
                reward[rew_index] += components["dribble_reward"][rew_index]

            # Cross effectiveness and position-based rewards for wingers
            if abs(o['ball'][1]) >= self.crossing_zones[0] and abs(o['ball'][1]) <= self.crossing_zones[1]:
                if o['sticky_actions'][2] or o['sticky_actions'][6]:  # action_top or action_bottom for crosses
                    components["cross_reward"][rew_index] = self.crossing_reward
                    reward[rew_index] += components["cross_reward"][rew_index]

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
