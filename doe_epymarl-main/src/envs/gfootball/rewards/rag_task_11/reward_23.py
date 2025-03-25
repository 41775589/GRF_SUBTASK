import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward based on offensive capabilities and fast-paced maneuvers."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle.get('sticky_actions', [0] * 10))
        return from_pickle

    def reward(self, reward):
        # Access the observation to compute additional rewards
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "aggressive_play_reward": [0.0] * len(reward)}
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        # Encourage fast-paced game and possession near the opponent's goal area
        for idx, o in enumerate(observation):
            if o['ball_owned_team'] == 0 and o['designated'] == o['active']:
                ball_x = o['ball'][0]  # Horizontal position of the ball
                # Reward agents for moving the ball forward beyond mid-field
                if ball_x > 0:
                    components['aggressive_play_reward'][idx] += 0.05 * (ball_x - 0.5)

                # Further reward for proximity to the opponent's goal
                if ball_x > 0.7:
                    components['aggressive_play_reward'][idx] += 0.1 * (ball_x - 0.7)
                    
                # Reward for dribbling towards the goal
                if o['sticky_actions'][9] == 1:  # action_dribble
                    components['aggressive_play_reward'][idx] += 0.02

            # Update the final reward for this player
            reward[idx] += components['aggressive_play_reward'][idx]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        
        # Append the details of reward components for monitoring
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        # Reset sticky actions count
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] += action_active
                info[f"sticky_actions_{i}"] = action_active

        return observation, reward, done, info
