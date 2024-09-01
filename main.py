import numpy as np
from agent import Agent
from policies import EpsilonGreedyPolicy

EPOCHS = 1000

# Initialize the agent and the policy
egreedypolicity = EpsilonGreedyPolicy(0.1)
agent = Agent(10, 10, egreedypolicity)

# Train the agent
for epoch in range(EPOCHS):
    action = agent.act()
    reward = np.random.randn() + action  # Reward is a random number with a bias towards the action taken
    agent.update(action, reward)

# Print the agent's action values (should be close to the true action values)
print(agent.action_values)
# >>> [-0.25129441  0.93802925  1.69312618  2.82281005  4.19756924  5.11255947 6.08260463  7.37916908  7.50885241  8.99611069]

# Print the agent's action counts (should be higher for actions with higher values)
print(agent.action_counts)
# >>> [ 11.  12.  11.  10.  15.   6.  18.   8.  12. 897.]
