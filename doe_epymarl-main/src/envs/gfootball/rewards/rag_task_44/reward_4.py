import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward focused on dribbling to a stop under pressure."""
    
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

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
        components = {"base_score_reward": reward.copy(), "dribble_stop_reward": [0.0] * len(reward)}

        for idx in range(len(reward)):
            o = observation[idx]            
            active = o['active']
            ball_owned_team = o['ball_owned_team']
            ball_owned_player = o['ball_owned_player']

            # Check if the active player currently owns the ball
            has_ball = (ball_owned_team == 0 and ball_owned_player == active)
            
            if has_ball:
                dribble_action_active = o['sticky_actions'][-2] == 1  # Last second to last in sticky actions is 'dribble'
                no_other_movement = np.sum(o['sticky_actions'][:8]) == 0  # No directional movements

                if dribble_action_active and no_other_movement:
                    components["dribble_stop_reward"][idx] = 0.1  # Reward stopping dribbling without moving
            reward[idx] += components["dribble_stop_reward"][idx]   # Add additional reward

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
                self.sticky_actions_counter[i] += action
        return observation, reward, done, info
