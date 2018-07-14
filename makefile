## Define variables
PY = python
BUILD_FLAGS = --inplace


## Define targets
all: build_ext

build_ext:
	$(PY) setup.py build_ext $(BUILD_FLAGS)

clean:
	$(RM) -rf build/
	find . -name "*.so" -delete
	for d in $(find . -name "__pycache__"); do echo $d; rm -rf $d; done

clean-dist:
	$(RM) -rf dist/ *.egg-info

clean-all: clean clean-dist

