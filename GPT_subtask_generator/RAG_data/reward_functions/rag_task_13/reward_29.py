import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a specialized defensive reward focused on the 'stopper' role."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = self.sticky_actions_counter.tolist()
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = np.array(from_pickle['CheckpointRewardWrapper'])
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": np.copy(reward), "defensive_actions_reward": [0.0, 0.0]}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)
        
        for i, o in enumerate(observation):
            if 'ball_owned_team' in o and o['ball_owned_team'] == 1:
                # Defensive reward is increased if our team (0) does not own the ball
                opponents_with_ball = np.where(o['right_team'] == o['ball_owned_player'])[0]

                if len(opponents_with_ball) > 0:
                    pos_opponent = o['right_team'][opponents_with_ball[0]]
                    pos_me = o['left_team'][o['active']]
                    dist = np.linalg.norm(pos_opponent - pos_me)

                    # Reward based on proximity to the ball-carrier from the opponent team
                    if dist < 0.05:
                        components['defensive_actions_reward'][i] += 0.2
                    elif dist < 0.1:
                        components['defensive_actions_reward'][i] += 0.1

            # Calculate final reward for this agent
            reward[i] += components['defensive_actions_reward'][i]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)

        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        if obs is not None:
            for agent_obs in obs:
                for i, action in enumerate(agent_obs['sticky_actions']):
                    if action:
                        self.sticky_actions_counter[i] += 1
                        info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, np.array(reward, dtype=np.float32), done, info
