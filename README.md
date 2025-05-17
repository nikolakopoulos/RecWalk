# RecWalk

Athanasios N. Nikolakopoulos and George Karypis. 2019. RecWalk: Nearly
Uncoupled Random Walks for Top-N Recommendation. In The Twelfth
ACM International Conference on Web Search and Data Mining (WSDM ’19),
February 11–15, 2019, Melbourne, VIC, Australia. ACM, New York, NY, USA,
9 pages. [pdf](http://nikolako.net/papers/ACM_WSDM2019_RecWalk.pdf)


## Example
We provide an example of both variations of RecWalk discussed in the paper. The code is written in Julia version 0.6 (an updated version that runs in current versions of Julia >= 1.0 is coming soon). 

For simplicity we also provide a split (target item per user alongside 99 randomly sampled unseen items (yahoo.mat) and a  corresponding item model (example.model). The item model can be built by solving the optimization problems per item described in Section 2.3.1 in the paper. For the example.model we use the SLIM software.    

## TODOS
Resolve issues to make the code compatible with Julia 1 (coming soon). 
Add a notebook with a more thorough example that includes other item models besides SLIM.  

