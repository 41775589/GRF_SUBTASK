import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds dense rewards for defensive behavior and quick transition to offensive plays."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)

        # Parameters for defensive behavior
        self.defense_weight = 1.0
        self.transition_weight = 0.5

        # Track players and ball possession
        self.previous_ball_owner = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.previous_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['CheckpointRewardWrapper'] = {
            'sticky_actions_counter': self.sticky_actions_counter,
            'previous_ball_owner': self.previous_ball_owner
        }
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_picle = self.env.set_state(state)
        self.sticky_actions_counter = from_picle['CheckpointRewardWrapper']['sticky_actions_counter']
        self.previous_ball_owner = from_picle['CheckpointRewardWrapper']['previous_ball_owner']
        return from_picle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "defensive_reward": [0.0] * len(reward),
                      "transition_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_owner = o.get("ball_owned_player")
            current_ball_team = o.get("ball_owned_team")

            # Check if defensive action was successful (ball was taken from opponent)
            if self.previous_ball_owner is not None and current_ball_team == 0 and self.previous_ball_owner == 1:
                components["defensive_reward"][rew_index] = self.defense_weight
                reward[rew_index] += components["defensive_reward"][rew_index]

            # Reward for transitioning to a quick counter-attack
            if (self.previous_ball_owner is None or self.previous_ball_owner == 1) and current_ball_team == 0:
                components["transition_reward"][rew_index] = self.transition_weight
                reward[rew_index] += components["transition_reward"][rew_index]

            # Update previous owner for the next step
            self.previous_ball_owner = current_ball_team

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
            for i, action in enumerate(agent_obs.get('sticky_actions', [])):
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
