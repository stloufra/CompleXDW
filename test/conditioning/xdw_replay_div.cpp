// Re-runs a worst case stored by xdw_binned_div in res/binned_results_div.csv.
//
// Usage: xdw_replay_div <bin 0..29> <combo 0..11>
//   bin   = decade of K: bin b covers K in [10^(b+1), 10^(b+2))
//   combo = index into DIV_COMBO_NAMES in src/combo_div.h

#include "binned_ops.h"
#include "binned_run.h"

int main(int argc, char** argv) { return replay_binned<DivOp>(argc, argv); }
