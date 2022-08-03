# Julia_Artifical_Fish_Swarm_Algorithm
Using AFSA to do binary classification in using Julia.

Parameters: 

1.Population size: the initial size of fish swarm

2.Visual: how far a fish can see, which can be understood as the radius of a circle.

3.Step: how far a fish can move each time. 

4.Crowding factor - Delta: the maximum number of fish in the field of visual

5.Number of repetitions - Try_number:  the iteration times.

6.Fitness: Accuracy

Key functions:


1.Prey: Find a random location within visual,
If its fitness better, move toward it:
Xnext=X + (Xv−X)/ |Xv−X| ∗ Step ∗ Rand()
Else return a random location within visual:
Xv=X + Visuanl ∗ Rand()


2.Swarm: Find the center point of all fish within,
If its fitness is better and not too crowded:
Xnext=X + (Xc−X)/ |Xv−X| ∗ Step ∗ Rand()
Else do Prey()


3.Follow: If X has a better and the neighbour is 
the best fish withini visual:
Xnext=X + (Xn−X)/ |Xv−X| ∗ Step ∗ Rand()
Else do Prey()




