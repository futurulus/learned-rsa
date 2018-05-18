#!/bin/sh
#
# Automatically generated with Qry (https://github.com/gangeli/qry)

# Restore working directory
mkdir -p ./runs/306/_rerun
# Run Program
'python'\
	'-u' 'sgd_literal_listener.py'\
	'--run_dir' './runs/306/_rerun/'\
	'--data_dir' 'singular/furniture'\
	'--l2_coeff' '0.01'\
	'--sgd_eta' '0.01'\
	'--generation' 'true'\
	'--features' 'cross_product'\
	'attr_type'\
	'attr_count'\
	'attr_pair'\
	'--cv' '5'\
	'--verbose' '1'