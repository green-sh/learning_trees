echo Add @profile as decorator for function that should be profiled
cd src/
PYTHONPATH=$PYTHONPATH:$PWD
kernprof -vl test/plot_regression_tree.py
