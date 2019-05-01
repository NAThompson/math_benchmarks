CXX := g++-8
CXXFLAGS := --std=gnu++17 -g -Wfatal-errors -Wall -Wextra -fvisibility=hidden
PERF := -fno-omit-frame-pointer -fno-unroll-loops -fno-peel-loops

LINKFLAGS := -L/usr/local/lib -pthread  -lbenchmark -lbenchmark_main -lquadmath
INC := -I../boost -I/usr/local/include


.PHONY: speed.x
speed.x: bench.cpp
	rm -f speed.x
	$(CXX) $(CXXFLAGS) -O3 -march=native $(INC) $< -o $@ $(LINKFLAGS)

.PHONY: clean
clean:
	rm -rf *.x *.x.dSYM perf.data perf.data.old
