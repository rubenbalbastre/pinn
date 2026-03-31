Let's solve an inverse PINN problem. The purpose of this notebook is to show that mathematical solutions can be discovered but they are not a physical solution. This can be seen in the results:

- we succesfully recover u(x,t)
- we do not recover alpha(x)

Essentially, what is hapenning is that the network u(x,t) learns everything alone so alpha(x) is not required to do anything (i.e: an horizontal line)