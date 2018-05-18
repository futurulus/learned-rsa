Code for Learning in the Rational Speech Acts Model
===================================================

Will Monroe, Christopher Potts
26 July 2016

You'll first need Python 2.7 with NumPy, as well as the TUNA corpus:

./get_tuna

To run the main model presented in the paper (last line of Table 1):

bash ./runs/308/_rerun.sh  # furniture
bash ./runs/309/_rerun.sh  # people

Check out the above shell scripts for the full command that runs the program
with tunable options.

Each of the "learned" models in Table 1 can be rerun with a similar command,
replacing the number in the run directory:

|------------|---------|------|
|Model       |Furniture|People|
|------------|---------|------|
|S0 basic    |     275 |  276 |
|S0 gen      |      11 |   12 |
|S0 basic+gen|     306 |  307 |
|            |         |      |
|S1 basic    |     251 |  252 |
|S1 gen      |       3 |    4 |
|S1 basic+gen|     308 |  309 |
|------------|---------|------|

Be prepared to wait a while for the models to finish! 2 days is normal for the
S1 models; the S0 ones should be faster. (Replicating results is great, but if
you'd rather not wait, see stdout.log in each run directory for the output of
my own run of the program.)
