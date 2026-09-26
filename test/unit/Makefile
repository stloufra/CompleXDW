# Unit tests.
#   make test                        build and run all
#   make test CXX=g++-16             another compiler (or set CXX in the environment)
#   make test OPT=-O3                optimisation level, default -O2
#   make test ARCH=                  target, default -march=native
#   make test CXXFLAGS="-g -Wall"    extra flags, appended last so they override
#   make exec/test_xdw               build one
#   MPFR_PREFIX=/path make test      MPFR outside Homebrew and the system paths
# Changing the compiler or any flag rebuilds everything.

TESTS := test_arith test_dw test_xdw test_xdw_span test_simd

OPT ?= -O2
ARCH ?= -march=native
REQUIRED := -std=c++20 -ffp-contract=off -fno-math-errno -I../..
MPFR_PREFIX ?= $(shell command -v brew > /dev/null && brew --prefix)
CPPFLAGS += $(if $(MPFR_PREFIX),-I$(MPFR_PREFIX)/include)
LDFLAGS += $(if $(MPFR_PREFIX),-L$(MPFR_PREFIX)/lib)
LDLIBS += -lmpfr -lgmp
FLAGS := $(strip $(OPT) $(ARCH) $(REQUIRED) $(CXXFLAGS))

STAMP := exec/.build-flags
$(shell mkdir -p exec; echo '$(CXX) $(CPPFLAGS) $(FLAGS)' | cmp -s - $(STAMP) || echo '$(CXX) $(CPPFLAGS) $(FLAGS)' > $(STAMP))

HEADERS := $(wildcard ../../*.h ../../src/*.h *.h)
BINARIES := $(addprefix exec/,$(TESTS))

ifeq ($(shell uname -s),Darwin)
CPU := $(shell sysctl -n machdep.cpu.brand_string)
else
CPU := $(shell grep -m1 'model name' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//')
endif

.PHONY: all test info clean

all: $(BINARIES)

exec/%: %.cpp $(HEADERS) $(STAMP)
	$(CXX) $(CPPFLAGS) $(FLAGS) $< $(LDFLAGS) $(LDLIBS) -o $@

info:
	@echo "Compiler  $$($(CXX) --version | head -n1)"
	@echo "CPU       $(or $(CPU),unknown)"
	@echo "Flags     $(FLAGS)"

# Runs every test even after a failure, then fails if any did.
test: info $(BINARIES)
	@bold= green= red= reset=; \
	if [ -t 1 ] && [ -z "$$NO_COLOR" ]; then bold='\033[1m' green='\033[32m' red='\033[31m' reset='\033[0m'; fi; \
	failed=""; \
	for t in $(TESTS); do \
		printf "\n$$bold==================== %s ====================$$reset\n" "$$t"; \
		./exec/$$t || failed="$$failed $$t"; \
	done; \
	echo; \
	if [ -n "$$failed" ]; then printf "$${red}FAILED:%s$$reset\n" "$$failed"; exit 1; fi; \
	printf "$${green}All %s unit tests passed (%s)$$reset\n" "$(words $(TESTS))" "$(CXX)"

clean:
	rm -rf exec
