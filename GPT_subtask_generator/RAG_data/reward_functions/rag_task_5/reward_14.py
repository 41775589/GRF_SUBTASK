import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that enhances rewards based on defensive actions to train immediate tactical responses and quick transitions for counter-attacks."""

    def __init__(self, env):
        super(CheckpointRewardWrapper, self).__init__(env)
        # This counter will keep track of special actions taken by each player
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        # Define the reward components
        self.goal_prevention_bonus = 0.1
        self.ball_steal_bonus = 0.1
        self.quick_transition_bonus = 0.05

    def reset(self):
        # Reset action counters along with the environment
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        # Stores additional state specific to the reward wrapper into the pickle state
        wrapped_state = self.env.get_state(to_pickle)
        wrapped_state['CheckpointRewardWrapper'] = {"StickyActionsCounter": self.sticky_actions_counter}
        return wrapped_state

    def set_state(self, state):
        # Retrieves additional state specific to the reward wrapper from the pickle state
        self.sticky_actions_counter = state.get('StickyActionsCounter', np.zeros(10, dtype=int))
        return self.env.set_state(state)

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "goal_prevention_bonus": [0.0] * len(reward),
                      "ball_steal_bonus": [0.0] * len(reward),
                      "quick_transition_bonus": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owned_team = o.get('ball_owned_team', -1)
            active_player = o.get('active', -1)
            sticky_actions = o.get('sticky_actions', [])

            if ball_owned_team == 0 and active_player == o['ball_owned_player']:
                # Calculate bonus for preventing goals (acting as defender)
                if o['left_team'][active_player][0] < -0.75:  # close to our goal
                    components["goal_prevention_bonus"][rew_index] = self.goal_prevention_bonus
                    reward[rew_index] += components["goal_prevention_bonus"][rew_index]

            if sticky_actions[9] == 1:  # if dribbling
                # Bonus for stealing the ball and initiating quick transitions
                components["ball_steal_bonus"][rew_index] = self.ball_steal_bonus
                reward[rew_index] += components["ball_steal_bonus"][rew_index]

            # Check for transitions (defensive action followed by a forward pass or move)
            if self.sticky_actions_counter[2] > 0 or self.sticky_actions_counter[4]:  # action top or right represent forward movement
                components["quick_transition_bonus"][rew_index] = self.quick_transition_bonus
                reward[rew_index] += components["quick_transition_bonus"][rew_index]

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
                self.sticky_actions_counter[action] += 1
                info[f"sticky_actions_{i}"] = action
        return observation, reward, done, info
