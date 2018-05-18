#!/bin/sh
#
# Automatically generated with Qry (https://github.com/gangeli/qry)

# Restore working directory
mkdir -p ._old/runs/4/_rerun
# Run Program
'python'\
	'-u' 'sgd_lsl.py'\
	'--run_dir' '._old/runs/4/_rerun/'\
	'--data_dir' 'singular/people'\
	'--generation' 'true'\
	'--features' 'null'\
	'attr_type'\
	'attr_count'\
	'attr_pair'\
	'--max_gen_length' '4'\
	'--sgd_eta' '0.01'\
	'--cv' '5'\
	'--sgd_max_iters' '10'\
	'--l2_coeff' '0.01'\
	'--sgd_use_adagrad' 'true'\
	'--samples_x' '10'\
	'--null_message' 'true'\
	'--only_relevant_alts' 'false'\
	'--verbose' '1'