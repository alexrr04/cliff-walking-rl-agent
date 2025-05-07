
def rollout(env, policy, max_steps=300):
    """
    Execute one episode with the greedy policy.

    Returns
    -------
    reached_goal : bool
    steps        : int
    total_return : float
    """
    state, _ = env.reset()
    total_return = 0.0
    for t in range(1, max_steps + 1):
        action = policy[state]
        state, reward, is_done, truncated, _ = env.step(action)
        total_return += reward

        if is_done:                        # reached [3,11]
            print(f"\n🎉  Goal reached in {t} steps, return = {total_return}\n")
            return True, t, total_return
        if truncated:                         # hit the TimeLimit wrapper
            break

    print("\n💥  Episode ended without reaching the goal\n")
    return False, t, total_return