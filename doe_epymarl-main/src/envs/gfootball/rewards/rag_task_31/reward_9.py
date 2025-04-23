import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds targeted defense rewards for tactical maneuvers."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.tackles_successful = 0
        self.sliding_effectiveness = 0
        # History keepers for past action efficiency
        self.history_tackles = []
        self.history_slides = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.tackles_successful = 0
        self.sliding_effectiveness = 0
        self.history_tackles = []
        self.history_slides = []
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Reward agents for successful tackles
            if 'tackle' in o['sticky_actions'] and o['sticky_actions']['tackle']:
                # Reward based on decrease in opponent's ball possession success
                if o['game_mode'] in [2, 3, 4]:  # Tackle during critical plays
                    components['defensive_reward'][rew_index] += 0.5
                else:
                    components['defensive_reward'][rew_index] += 0.2
                self.tackles_successful += 1

            # Reward agents for effective sliding
            if 'slide' in o['sticky_actions'] and o['sticky_actions']['slide']:
                # Evaluate the effect of slide: did it change ball possession?
                if o['ball_owned_team'] == 0:  # Assuming agent's team is 0
                    components['defensive_reward'][rew_index] += 0.3
                else:
                    components['defensive_reward'][rew_index] += 0.1
                self.sliding_effectiveness += 1

            # Aggregate the rewards
            reward[rew_index] += components['defensive_reward'][rew_index]

        # Update histories for adaptation in longer sessions
        self.history_tackles.append(self.tackles_successful)
        self.history_slides.append(self.sliding_effectiveness)
        
        return reward, components

    def get_state(self, to_pickle):
        to_pickle['defensive_history'] = (self.history_tackles, self.history_slides)
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.history_tackles, self.history_slides = from_pickle['defensive_history']
        return from_pickle

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        # Update sticky actions for record-keeping
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
                
        return observation, reward, done, info
