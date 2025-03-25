import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward structure focusing on the sweeper role. This rewards clearing the ball from
    the defensive zone, executing key tackles, and supporting the defense through quick positional recoveries."""

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearances = {}
        self.tackles = {}

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.clearances = {}
        self.tackles = {}
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['clearances'] = self.clearances
        to_pickle['tackles'] = self.tackles
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.clearances = from_pickle.get('clearances', {})
        self.tackles = from_pickle.get('tackles', {})
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "clearance_reward": [0.0] * len(reward),
                      "tackle_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            active_player_pos = o['left_team'][o['active']] if o['active'] >= 0 else None
            
            # Check if the player made a clearance: ball moving away from goal area when in possession
            if active_player_pos is not None and o['ball_owned_team'] == 0:
                if active_player_pos[0] < -0.5 and np.linalg.norm(o['ball_direction'][:2]) > 0.1:
                    if rew_index not in self.clearances:
                        components["clearance_reward"][rew_index] = 0.3
                        reward[rew_index] += components["clearance_reward"][rew_index]
                        self.clearances[rew_index] = True

            # Reward tackles: Losing ball possession near own goal without opponent scoring
            if active_player_pos is not None and o['ball_owned_team'] == -1:
                previous_owner = self.env.unwrapped.previous_control()
                if previous_owner and previous_owner == rew_index:
                    if active_player_pos[0] < -0.5:
                        if rew_index not in self.tackles:
                            components["tackle_reward"][rew_index] = 0.2
                            reward[rew_index] += components["tackle_reward"][rew_index]
                            self.tackles[rew_index] = True

        return reward, components

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward, components = self.reward(reward)
        info["final_reward"] = sum(reward)   # Final composite reward after applying all components
        for key, value in components.items():
            info[f"component_{key}"] = sum(value)
        obs = self.env.unwrapped.observation()
        self.sticky_actions_counter.fill(0)
        for agent_obs in obs:
            for i, action in enumerate(agent_obs['sticky_actions']):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
