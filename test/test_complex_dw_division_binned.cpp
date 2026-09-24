// Long-running complex DW division conditioning sweep, meant for a multi-day cluster job.
// Checkpoints to res/binned_results_div.csv and res/binned_log_div.txt, see src/binned_run.h.

#include "src/binned_ops.h"
#include "src/binned_run.h"

int main() { return run_binned<DivOp>(); }
