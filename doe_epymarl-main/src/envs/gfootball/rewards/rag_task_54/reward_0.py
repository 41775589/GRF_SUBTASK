import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """
    A wrapper that adds a reward for collaborative plays between shooters and passers.
    This involves detecting sequences where a controlled player successfully passes the ball resulting in a goal.
    """
    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_owner = None
        self.pass_to_shoot_reward = 0.3

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.last_ball_owner = None
        return self.env.reset()

    def get_state(self, to_pickle):
        state = self.env.get_state(to_pickle)
        state['last_ball_owner'] = self.last_ball_owner
        return state

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.last_ball_owner = from_pickle.get('last_ball_owner', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()
        components = {"base_score_reward": reward.copy(),
                      "pass_to_shoot_reward": [0.0] * len(reward)}

        if observation is None:
            return reward, components

        assert len(reward) == len(observation)

        for rew_index in range(len(reward)):
            o = observation[rew_index]
            current_ball_owner = (
                o['ball_owned_team'], o['ball_owned_player']
            ) if o['ball_owned_team'] in [0, 1] else None

            if reward[rew_index] == 1 and self.last_ball_owner is not None and current_ball_owner != self.last_ball_owner:
                # A goal was scored and the ownership of the ball changed in the build-up,
                # which often is an indicator of a pass
                diff_team = self.last_ball_owner[0] != current_ball_owner[0]
                if not diff_team:  # Same team, potential pass-to-shoot sequence
                    components["pass_to_shoot_reward"][rew_index] = self.pass_to_shoot_reward
                    reward[rew_index] += components["pass_to_shoot_reward"][rew_index]

            # Update the ball owner tracker
            if current_ball_owner and o['game_mode'] == 0: # Normal play mode
                self.last_ball_owner = current_ball_owner

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
