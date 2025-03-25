import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances the training of agents focusing on offensive capabilities,
       particularly fast-paced attacking maneuvers and precision-based finishing."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Initialize the reward for entering each advancing field zone
        self.advance_reward = 0.05
        # Initialize reward for successful dribbles near opponent's goal
        self.dribble_reward = 0.15
        # Track number of dribbles in attacking third
        self.dribble_counters = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.dribble_counters = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['dribble_counters'] = self.dribble_counters
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state = self.env.set_state(state)
        self.dribble_counters = state.get('dribble_counters', {})
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {
            "base_score_reward": reward.copy(),
            "advance_reward": [0.0] * len(reward),
            "precision_dribble_reward": [0.0] * len(reward)
        }

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for idx, obs in enumerate(observation):
            # Encourage forward progression with the ball towards the opponent's goal
            if obs['ball_owned_team'] == 0 and obs['active'] == obs['ball_owned_player']:
                x_ball, y_ball = obs['ball'][0], obs['ball'][1]
                
                # Advanced positions on the field are rewarded
                if x_ball > 0.5:  # Ball in the opponent's half
                    components['advance_reward'][idx] = self.advance_reward
                    reward[idx] += components['advance_reward'][idx]

                # Encourage keeping the ball under control and dribbling in advanced areas
                if x_ball > 0.7 and np.any(obs['sticky_actions'][9]):  # Dribbling in attacking third
                    dribbles = self.dribble_counters.get(idx, 0)
                    if dribbles < 3:  # Limit the dribble rewards, e.g., up to 3 per possession
                        components['precision_dribble_reward'][idx] = self.dribble_reward
                        reward[idx] += components['precision_dribble_reward'][idx]
                        self.dribble_counters[idx] = dribbles + 1
            
        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        reward, components = self.reward(reward)
        
        info['final_reward'] = sum(reward)
        for key, value in components.items():
            info[f'component_{key}'] = sum(value)
        
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action_active in enumerate(agent_obs['sticky_actions']):
                if action_active:
                    self.sticky_actions_counter[i] += 1
                info[f'sticky_actions_{i}'] = self.sticky_actions_counter[i]

        return observation, reward, done, info
