import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dribbling and direction changing reward component focused on 1v1 situations against the goalkeeper."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter.copy()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        return from_pickle

    def reward(self, rewards):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": rewards.copy(),
                      "dribbling_reward": [0.0] * len(rewards)}
        if observation is None:
            return rewards, components

        assert len(rewards) == len(observation)

        for idx in range(len(rewards)):
            o = observation[idx]
            base_reward = rewards[idx]

            # Encourage maintaining possession under pressure
            if o['ball_owned_team'] == 0 and o['active'] == o['ball_owned_player']:
                # Close to the opponent's goal and navigating
                if o['ball'][0] > 0.5:  # Ball is on opponent's half
                    dribbling_effectiveness = np.sum(o['sticky_actions'][8:10])  # Dribbling or sprinting 
                    components["dribbling_reward"][idx] = dribbling_effectiveness * 0.05
                    rewards[idx] += components["dribbling_reward"][idx]
                    
                # High reward for feints and direction changes near the goalkeeper
                if np.abs(o['ball'][1]) < 0.1 and o['ball'][0] > 0.9:
                    # Dribble or change direction in front of the goalkeeper
                    if o['sticky_actions'][8] == 1 or (o['sticky_actions'][0:8] != self.sticky_actions_counter[0:8]).any():
                        rewards[idx] += 0.2  # bonus for effective feint/direction change

            # Store current sticky actions for next comparison
            self.sticky_actions_counter = o['sticky_actions']

        return rewards, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        return observation, reward, done, info
