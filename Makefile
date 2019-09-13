PYTHON = python3

.PHONY: clean run view

clean :
	- \rm -R __pycache__
	- \rm -f *.pyc 
