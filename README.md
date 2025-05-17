# RecWalk

Athanasios N. Nikolakopoulos and George Karypis. 2019. RecWalk: Nearly
Uncoupled Random Walks for Top-N Recommendation. In The Twelfth
ACM International Conference on Web Search and Data Mining (WSDM ’19),
February 11–15, 2019, Melbourne, VIC, Australia. ACM, New York, NY, USA,
9 pages. [pdf](http://nikolako.net/papers/ACM_WSDM2019_RecWalk.pdf)


## Example
We provide an example of both variations of RecWalk discussed in the paper. The repository now includes a Python implementation (`recwalk` package) in addition to the original Julia code. The Python version requires Python 3.8+ along with `numpy` and `scipy`.

For simplicity we also provide a split (target item per user alongside 99 randomly sampled unseen items (yahoo.mat) and a  corresponding item model (example.model). The item model can be built by solving the optimization problems per item described in Section 2.3.1 in the paper. For the example.model we use the SLIM software.    

## TODOS
Add a notebook with a more thorough example that includes other item models besides SLIM.  


### Running the Python example

After installing the required dependencies you can run `example.py` to reproduce
the results using the Python implementation:

```bash
pip install -r requirements.txt  # installs numpy and scipy
python example.py
```
