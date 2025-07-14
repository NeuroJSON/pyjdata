
PY=python3

all: pretty test build
pretty:
	$(PY) -m black test/*.py jdata/*.py setup.py

test:
	$(PY) -m unittest discover -v test

build:
	$(PY) -m build

report:
	@echo '====== all imported functions ======'
	@grep '^\s*[a-z]*,' jdata/__init__.py | sed -e 's/^\s*//g' -e 's/,//g' | sort | uniq -c
	@echo '====== all tested functions ======'
	@grep 'def\s*test' test/run_test.py | sed -e 's/\s*def test_//g' -e 's/(self)://g' -e 's/_.*//g' | sort | uniq -c

.DEFAULT_GOAL=all
.PHONY: all pretty test report
