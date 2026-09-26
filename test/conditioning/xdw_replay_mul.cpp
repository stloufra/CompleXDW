// Re-runs a worst case stored by xdw_binned_mul in res/binned_results_mul.csv.
//
// Usage: xdw_replay_mul <bin 0..29> <combo 0..5>
//   bin   = decade of K: bin b covers K in [10^(b+1), 10^(b+2))
//   combo = index into COMBO_NAMES in src/combo_mul.h

#include "binned_ops.h"
#include "binned_run.h"

int main(int argc, char** argv) { return replay_binned<MulOp>(argc, argv); }
