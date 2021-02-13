dev_branch := dev

help :
	cat Makefile

install :
	pip install -e .

dev :
	git checkout $(dev_branch)
	git status

commit_everything : dev
	git status
	git add .
	git commit

test :
	pytest

test_forest_fire : 
	pytest gym_cellular_automata/tests/envs/forest_fire/

.PHONY: help install test test_forest_fire dev
