import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a dense reward for defensive actions and quick transitions.
    It measures success by the player's ability to interrupt opponent progress and initiate counter-attacks.
    """

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_possession_change_counter = [0, 0]

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.ball_possession_change_counter = [0, 0]
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['sticky_actions'] = self.sticky_actions_counter
        to_pickle['possession_changes'] = self.ball_possession_change_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.sticky_actions_counter = from_pickle['sticky_actions']
        self.ball_possession_change_counter = from_pickle['possession_changes']
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Award for ball possession change from opponent to agent.
            if o['ball_owned_team'] != self.ball_possession_change_counter[rew_index]:
                if o['ball_owned_team'] == 0:  # Ball now owned by the agent's team
                    components["defensive_reward"][rew_index] += 0.5
                    reward[rew_index] += 0.5
                self.ball_possession_change_counter[rew_index] = o['ball_owned_team']
            
            # Award for moving the ball to opponent's half quickly if possession just gained.
            if components["defensive_reward"][rew_index] > 0 and o['ball'][0] > 0.5:
                components["transition_reward"][rew_index] = 0.3
                reward[rew_index] += 0.3

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
