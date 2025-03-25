import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for tactical defensive actions and counter-attacks."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.recoveries = 0
        self.quick_transitions = 0
        self.reward_for_recovery = 0.3
        self.reward_for_transition = 0.5

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.recoveries = 0
        self.quick_transitions = 0
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['recoveries'] = self.recoveries
        to_pickle['quick_transitions'] = self.quick_transitions
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.recoveries = from_pickle['recoveries']
        self.quick_transitions = from_pickle['quick_transitions']
        return from_pickle

    def reward(self, reward):
        # Extract observation from the environment.
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "recovery_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}
        
        if observation is None:
            return reward, components
        
        assert len(reward) == len(observation)
        
        for rew_index in range(len(reward)):
            o = observation[rew_index]
            
            # Defensive recovery: if the ball ownership changes to the current agent's team
            if o['ball_owned_team'] == 0:  # Assuming the agent's team is indexed as 0
                components["recovery_reward"][rew_index] = self.reward_for_recovery
                reward[rew_index] += 1.5 * components["recovery_reward"][rew_index]
                self.recoveries += 1
            
            # Quick transition: if quickly moving from defense to a scoring opportunity
            if o['game_mode'] == 3:  # Assuming game mode 3 indicates a quick transition opportunity
                components["transition_reward"][rew_index] = self.reward_for_transition
                reward[rew_index] += 1.5 * components["transition_reward"][rew_index]
                self.quick_transitions += 1
                
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
            for i, active in enumerate(agent_obs['sticky_actions']):
                self.sticky_actions_counter[i] = active
        return observation, reward, done, info
