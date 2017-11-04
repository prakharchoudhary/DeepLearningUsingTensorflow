import gym
import numpy as np

environment = gym.make('FrozenLake-v0')
S = environment.observation_space.n
A = environment.action_space.n

#Initialize table with all zeros
Q = np.zeros([S,A])

# Set learning parameters
alpha = .85
gamma = .99
num_episodes = 2000

#create lists to contain total rewards and steps per episode
rList = []

for i in range(num_episodes):
	#Reset environment and get first new observation
	s = environment.reset()
	cummulative_reward = 0
	d = False
	j = 0

	# Q-Learning Algorithm
	while j < 99:
		j += 1
		#Choose an action by greedily (with noise) picking from Q table
		a = np.argmax(Q[s,:]+\
		              np.random.randn(1, environment.action_space.n)\
		              *(1./(i+1)))

		# Get new state and reward from environment
		s1,r,d,_ = environment.step(a)
		#Update Q-Table with new knowledge
		Q[s,a] = Q[s,a] + alpha*(r + gamma*np.max(Q[s1,:]) - Q[s,a])
		cummulative_reward += r
		s = s1
		if d == True:
			break
	rList.append(cummulative_reward)

print "Score over time: " + str(sum(rList)/num_episodes)
print "Final Q-Table Values"
print Q