import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that applies a dynamic reward structure focusing on offensive strategies,
    optimizing team coordination, and adapting between scoring and positioning strategies.
    """
    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_position = None
        self.prev_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_position = None
        self.prev_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'prev_ball_position': self.prev_ball_position,
            'prev_ball_owner': self.prev_ball_owner
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        state_data = from_pickle['CheckpointRewardWrapper']
        self.prev_ball_position = state_data['prev_ball_position']
        self.prev_ball_owner = state_data['prev_ball_owner']
        return from_pickle

    def reward(self, reward):
        obs = self.env.unwrapped.observation()
        if obs is None:
            return reward, {}

        components = {"base_score_reward": reward.copy()}
        
        for i in range(len(reward)):
            o = obs[i]
            base_reward = reward[i]
            
            # Reward strategy adaptation (positional play and scoring)
            if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                # Enhance reward for smart positioning and passing plays
                if self.prev_ball_owner is not None and self.prev_ball_owner != o['active']:
                    # Reward passed ball control within team
                    base_reward += 0.5
                else:
                    # Reward maintaining possession under pressure
                    base_reward += 0.1
                
            # Score goal: High reward
            if 'score' in o:
                if o['score'][1] > (self.prev_score[1] if self.prev_score else 0):
                    base_reward += 2  # rewarded more for scoring a goal

            # Update for potential next steps
            self.prev_ball_position = o['ball']
            self.prev_ball_owner = o['ball_owned_player'] if o['ball_owned_team'] == 1 else None
            self.prev_score = o['score']
            
            reward[i] = base_reward

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
