
PY=python3

all: pretty test build
pretty:
	$(PY) -m black test/*.py jdata/*.py setup.py

test:
	$(PY) -m unittest discover -v test

build:
	$(PY) -m build


.DEFAULT_GOAL=all
.PHONY: all pretty test
