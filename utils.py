import numpy as np

def policy(s, q, actions, p_highest, p_other):
    """epsilon-greedy policy as implied through the state-action values.
    
        For a given state, the policy is to choose the action with the highest state-action
        value with probability 1 - epislion + (epsilion / |actions|). The remaining actions
        are selected with probability epsilion / |actions|.
    
    """
    
    # Get state-action values for the current state
    state_action_values = [q[s + (a,)] for a in range(4)]

    # The best action based on state-action values
    best_actions = np.argwhere(state_action_values == np.amax(state_action_values))[:, 0]

    # If tie-breaks exist choose an action randomly
    if best_actions.shape[0]>1:
        a = np.random.choice(best_actions)
    else:
        a = best_actions[0]

    # Set the probabilities for random choice
    probs = [p_other] * 4
    probs[a] = p_highest

    return np.random.choice(actions, p=probs)

def update_state_action_values(returns, counts, q):
    """Updates state-action values."""

    # Update state action values with means
    for k in returns.keys():
        q[k] = returns[k] / counts[k]

    return q

def play_episode_from_policy(env, q, n_trials, epsilon=0.15, render=False):
    """Renders an episode using state action values and epsilon-greedy policy.
       Returns success rate. This function is for debugging and not used in the graded 
       assignment."""
    
    actions = [0, 1, 2, 3]
    p_highest = 1 - epsilon + epsilon/4.0
    p_other = epsilon/4.0
    successes = 0
    failures = 0
    for t in range(n_trials):
        env.reset()
        s_0 = env.state
        terminate = False
        while terminate is False:
            if render:
                env.render()
            a_0 = policy(s_0, q, actions, p_highest, p_other)
            s_1, reward, terminate_temp = env.step(a_0)

            # Add to success and failures
            if reward==9:
                successes+=1
            if reward==-11:
                failures+=1

            # Set terminate variable - done here so we don't exit the loop
            #  before recording successes and failures
            terminate = terminate_temp

            # current state ← new state
            s_0 = s_1
        if render:
            env.render()

    return np.round(successes/(successes + failures), 4)

def softmax(logits):
    exps = np.exp(logits - np.max(logits))
    return exps / np.sum(exps)