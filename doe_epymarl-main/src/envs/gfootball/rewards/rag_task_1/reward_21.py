import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that modifies rewards based on agents' offensive maneuvers and game phases."""
    
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.active_phase_rewards = {
            'NORMAL': 0.01,          # small reward during normal play
            'KICKOFF': 0.2,          # higher reward during kickoff to promote quick playmaking
            'CORNERS': 0.3,          # reward for earning corner kicks suggests pushing towards the goal
            'FREE_KICK': 0.25        # reward for earning free kicks in offensive half
        }

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions_counter'] = self.sticky_actions_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        state = self.env.set_state(state)
        self.sticky_actions_counter = state['sticky_actions_counter']
        return state

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy()}
        
        new_rewards = [0.0] * len(reward)

        if observation is None:
            return reward, components

        for idx, o in enumerate(observation):
            components.setdefault('offensive_play_reward', []).append(0.0)
            # Check if it is a good attack phase
            if o['game_mode'] in self.active_phase_rewards:
                # Check proximity to opponent goal
                if o['ball'][0] > 0:  # Ball is on opponent's half
                    components['offensive_play_reward'][idx] = self.active_phase_rewards[o['game_mode']]
                    new_rewards[idx] += components['offensive_play_reward'][idx]
                
            # Sum all the rewards
            reward[idx] += new_rewards[idx]

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
