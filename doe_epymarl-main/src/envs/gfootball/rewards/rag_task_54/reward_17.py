import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a collaborative reward focusing on enhancing the effectiveness of plays 
    between shooters and passers to exploit scoring opportunities fully.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.passing_bonus = 0.1
        self.shooting_bonus = 0.2

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['sticky_actions_counter'] = self.sticky_actions_counter
        return state

    def set_state(self, state):
        self.sticky_actions_counter = state.get('sticky_actions_counter', np.zeros(10, dtype=int))
        self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "passing_bonus": [0.0] * len(reward),
                      "shooting_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index, o in enumerate(observation):
            base_reward = reward[rew_index]
            if 'ball_owned_team' not in o:
                continue
            
            if o['ball_owned_team'] == 0:  # Ball owned by the left team
                
                # Encourage passing
                if 'sticky_actions' in o and o['sticky_actions'][9] == 1:  # Check if dribble action is active
                    components['passing_bonus'][rew_index] = self.passing_bonus
                    reward[rew_index] += components['passing_bonus'][rew_index]
                
                # Encourage shooting
                if np.linalg.norm(o['ball_direction'][0:2]) > 0.01:  # Significant ball movement towards the goal
                    components['shooting_bonus'][rew_index] = self.shooting_bonus
                    reward[rew_index] += components['shooting_bonus'][rew_index]

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)

        # Update sticky actions counter
        obs = self.env.unwrapped.observation()
        for agent_obs in obs:
            for i, action_status in enumerate(agent_obs['sticky_actions']):
                if action_status == 1:
                    self.sticky_actions_counter[i] += 1
                info[f"sticky_actions_{i}"] = self.sticky_actions_counter[i]

        return observation, reward, done, info
