c:
	@python3 setup.py build_ext -i

clean:

	@rm -f src/hplop/**/*cpython*
	@rm -f src/hplop/**/*.html
	@rm -rf src/hplop/**/__pycache__
	@rm -f src/hplop/**/*.c
	@rm -rf src/demo/__pycache__
