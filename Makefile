
PYTHON_PATH="/opt/conda/envs/cprl/bin/python"

tsptw:
	rm -rf src/problem/tsptw/solving/build
	cmake -Hsrc/problem/tsptw/solving -Bsrc/problem/tsptw/solving/build -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH) -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING="Release" -G "Unix Makefiles" -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)
	cmake --build src/problem/tsptw/solving/build --config Release --target all -- -j $(nproc) VERBOSE=1
	# mv src/problem/tsptw/solving/build/solver_tsptw ./
	mv src/problem/tsptw/solving/build/solver_tsptw.cpython-37m-x86_64-linux-gnu.so ./

portfolio:
	rm -rf src/problem/portfolio/solving/build
	cmake -Hsrc/problem/portfolio/solving -Bsrc/problem/portfolio/solving/build -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH) -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING="Release" -G "Unix Makefiles" -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)
	cmake --build src/problem/portfolio/solving/build --config Release --target all -- -j $(nproc) VERBOSE=1
	mv src/problem/portfolio/solving/build/solver_portfolio ./

