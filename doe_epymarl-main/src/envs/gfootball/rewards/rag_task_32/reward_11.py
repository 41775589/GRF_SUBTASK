import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that encourages wingers to practice sprinting and crossing accurately."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.crossing_zones = [(-1, 0.4), (-0.8, 0.42), (0.8, 0.42), (1, 0.4)]  # Zones in the corners for crosses
        self.crossing_reward = 0.5  # Reward for accurate crossing
        self.sprint_reward = 0.3   # Reward for initiating sprints towards crossing zones

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
                      "crossing_reward": 0.0,
                      "sprint_reward": 0.0}

        if observation is None:
            return reward, components
        
        for ing_index, o in enumerate(observation):
            # Check for sprint action initialization
            if o['sticky_actions'][8]:
                self.sticky_actions_counter[8] += 1
                if self.sticky_actions_counter[8] == 1:  # Reward only on the first tick of the action
                    components["sprint_reward"] += self.sprint_reward
                    reward[ing_index] += components["sprint_reward"]

            # Check for successful cross into designated zones
            ball_position = o['ball'][:2]
            if any([np.linalg.norm(np.array(ball_position) - np.array(zone)) < 0.1 for zone in self.crossing_zones]):
                if o['ball_owned_player'] == o['active']:
                    components["crossing_reward"] += self.crossing_reward
                    reward[ing_index] += components["crossing_reward"]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = value
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
