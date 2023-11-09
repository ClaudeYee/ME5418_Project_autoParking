class RolloutBuffer():
    def __init__(self):
        self.states = []
        self.episode_lengths = []

    def rollout(self, states, episode_length, t):
        if t <= TIMESTEPS_ROLLOUT:
            self.add_data(states, action_indices, log_probs, valid_actions, rewards, episode_length)
        else:
            end = TIMESTEPS_ROLLOUT - sum(self.episode_lengths)
            index = range(end)
            index_tmp = range(episode_length - end)

            self.add_data(states(index), action_indices(index), log_probs(index),
                          valid_actions(index), rewards(index), end)

            tmp_buffer = RolloutBuffer()
            tmp_buffer.add_data(states(index_tmp), action_indices(index_tmp), log_probs(index_tmp),
                                valid_actions(index_tmp), rewards(index_tmp), episode_length - end)
            return tmp_buffer

        return None

    def add_data(self, states, action_indices, log_probs, valid_actions, rewards, episode_length):
        self.states.append(states)
        self.action_indices.append(action_indices)
        self.log_probs.append(log_probs)
        self.valid_actions.append(valid_actions)
        self.rewards.append(rewards)
        self.episode_lengths.append(episode_length)
