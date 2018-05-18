#!/bin/sh
#
# Automatically generated with Qry (https://github.com/gangeli/qry)

# Restore working directory
mkdir -p ./runs/276/_rerun
# Run Program
'python'\
	'-u' 'sgd_literal_listener.py'\
	'--run_dir' './runs/276/_rerun/'\
	'--data_dir' 'singular/people'\
	'--l2_coeff' '0.01'\
	'--sgd_eta' '0.01'\
	'--generation' 'true'\
	'--cv' '5'\
	'--verbose' '1'