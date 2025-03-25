import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a reward for successful sliding tackles, emphasizing timing and precision."""

    def __init__(self, env):
        super().__init__(env)
        self.previous_ball_owner = None
        self.tackle_success_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)

    def reset(self):
        self.previous_ball_owner = None
        self.tackle_success_counter = 0
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['previous_ball_owner'] = self.previous_ball_owner
        to_pickle['tackle_success_counter'] = self.tackle_success_counter
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.previous_ball_owner = from_pickle['previous_ball_owner']
        self.tackle_success_counter = from_pickle['tackle_success_counter']
        return from_pickle

    def reward(self, reward):
        # Process observation to give rewards based on successful tackles
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "tackle_reward": [0] * len(reward)}
        
        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            ball_owner_team = o.get('ball_owned_team')
            active_player_team = 0 if o['active'] < len(o['left_team']) else 1

            # Tackle logic: If the ball ownership switches to the opponent due to a tackle
            if 'sticky_actions' in o:
                is_tackle_action = o['sticky_actions'][9]  # Assuming index '9' for "action_sliding"
                just_tackled = (self.previous_ball_owner is not None and
                                self.previous_ball_owner != ball_owner_team and
                                ball_owner_team != -1 and
                                active_player_team != ball_owner_team)

                if is_tackle_action and just_tackled:
                    # Reward is higher for successful tackles
                    components['tackle_reward'][rew_index] = 0.5  # Reward value can be tuned for balancing
                    self.tackle_success_counter += 1

                self.previous_ball_owner = ball_owner_team

            reward[rew_index] += components['tackle_reward'][rew_index]

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
