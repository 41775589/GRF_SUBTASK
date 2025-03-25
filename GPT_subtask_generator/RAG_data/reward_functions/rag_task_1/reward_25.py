import gym
import numpy as np
class CheckpointRewardWrapper(gym.RewardWrapper):
    """A wrapper that adds a dense reward for mastering offensive maneuvers and quick attack responses."""

    def __init__(self, env):
        super().__init__(env)
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.aggressive_play_reward = 0.2
        self.dynamic_adaptation_reward = 0.15
        self.prev_ball_position = None

    def reset(self):
        self.sticky_actions_counter = np.zeros(10, dtype=int)
        self.prev_ball_position = None
        return self.env.reset()

    def get_state(self, to_pickle):
        to_pickle['prev_ball_position'] = self.prev_ball_position
        return self.env.get_state(to_pickle)

    def set_state(self, state):
        from_pickle = self.env.set_state(state)
        self.prev_ball_position = from_pickle.get('prev_ball_position', None)
        return from_pickle

    def reward(self, reward):
        observation = self.env.unwrapped.observation()

        if observation is None:
            return reward, {}

        components = {'base_score_reward': reward.copy()}
        
        aggressive_rewards = []
        dynamic_adaptation_rewards = []

        for rew_index in range(len(reward)):
            o = observation[rew_index]

            # Encourage aggressive forward movement if controlled by the agent.
            if o['active'] != -1:
                # Calculate the reward for advancing the ball closer to opponent's goal.
                if o['ball_owned_team'] == 0 and self.prev_ball_position is not None:
                    x_progress = o['ball'][0] - self.prev_ball_position[0]
                    if x_progress > 0:  # Forward movement toward the opponent's goal.
                        aggressive_rewards.append(self.aggressive_play_reward * x_progress)
                    else:
                        aggressive_rewards.append(0)
                else:
                    aggressive_rewards.append(0)
                
                # Encourage dynamic movement adaptations based on game context.
                if o['game_mode'] not in [0, 1]:
                    # Game is in a special mode (corner, free-kick)
                    if o['ball_owned_team'] == 1 and o['active'] == o['ball_owned_player']:
                        dynamic_adaptation_rewards.append(self.dynamic_adaptation_reward)
                    else:
                        dynamic_adaptation_rewards.append(0)
                else:
                    dynamic_adaptation_rewards.append(0)
                
            else:
                aggressive_rewards.append(0)
                dynamic_adaptation_rewards.append(0)

            self.prev_ball_position = o['ball']

        # Aggregate and apply rewards.
        for i in range(len(reward)):
            reward[i] += aggressive_rewards[i] + dynamic_adaptation_rewards[i]

        components['aggressive_play_reward'] = aggressive_rewards
        components['dynamic_adaptation_reward'] = dynamic_adaptation_rewards

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
