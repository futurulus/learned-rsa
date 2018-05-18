#!/bin/sh
#
# Automatically generated with Qry (https://github.com/gangeli/qry)

# Restore working directory
mkdir -p ._old/runs/11/_rerun
# Run Program
'python'\
	'-u' 'sgd_literal_listener.py'\
	'--run_dir' '._old/runs/11/_rerun/'\
	'--data_dir' 'singular/furniture'\
	'--generation' 'true'\
	'--features' 'null'\
	'attr_type'\
	'attr_count'\
	'attr_pair'\
	'--sgd_eta' '0.01'\
	'--cv' '5'\
	'--sgd_max_iters' '10'\
	'--l2_coeff' '0.01'\
	'--verbose' '1'