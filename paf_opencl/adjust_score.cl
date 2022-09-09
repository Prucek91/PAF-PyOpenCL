//#pragma OPENCL EXTENSION cl_intel_printf : enable

int idx_(int i, int j, int N) { return (j * N) + i; }

char SOF(int i, int j, int k, __global char *F, int N) {
  if (i < N && j < N) {
    int idx_i = j + 1;
    int idx_j = min(k, (2 * j) - i + 1);
    return F[idx_(idx_i, idx_j, N)];
  } else {
    return 0;
  }
}

char S(int i, int j, __global char *F, int N) {
  if (j > N) {
    return 0;
  }

  char result = 0;
  int k = j + 1;
  char result1 = SOF(i, j, k, F, N) + S(j + 1, k, F, N);
  if (result1 > result) {
    result = result1;
  }
  return result;
}

__kernel void adjust_score(__global char *c, __global char *F, const int N) {
  int x = get_global_id(0); // x
  int y = get_global_id(1); // y
  // c[(y * N) + x] = (y * N )+ x; - index 2D w macierzy 1D
  int i = x;
  int j = y;

  // podzial zadan
  const int grid_size = 128;
  int idx_per_thread = N / grid_size;

  float t = (N / grid_size);
  int MIN_BOUND = floor(t);
  int MAX_BOUND = ceil(t);
  int MIN_X_IDX = i * MIN_BOUND;
  int MAX_X_IDX = (i * MAX_BOUND) + MAX_BOUND;
  int MIN_Y_IDX = j * MIN_BOUND;
  int MAX_Y_IDX = (j * MAX_BOUND) + MAX_BOUND;

  // printf("Kernel i %d j %d zakres %d - %d, %d - %d \n", x, y, MIN_X_IDX,
  // MAX_X_IDX, MIN_Y_IDX,MAX_Y_IDX);

  for (int ii = MIN_X_IDX; ii <= MAX_X_IDX; ii++) {
    for (int jj = MIN_Y_IDX; jj <= MAX_Y_IDX; jj++) {
      i = ii;
      j = jj;

      int idx = idx_(i, j, N);
      if (idx >= (N * N)) {
        c[idx] = 0;
      } else {

        // zredukowany wzor rekurencyjny 3.2
        if (j >= (N - 1)) {
          c[idx] = 0;
        } else {
          char result = 0;

          for (int k = j + 1; k <= N; k++) {
            char tmp = 0;
            int idx_F_i = j + 1;
            int idx_F_j = min(k, (2 * j) - i + 1);
            int idx_S_i = j + 1;
            int idx_S_j = k;
            tmp = F[idx_(idx_F_i, idx_F_j, N)] + c[idx_(idx_S_i, idx_S_j, N)];
            if (result < tmp) {
              result = tmp;
            }
          }

          c[idx] = result;
        }
      }
    }
  }
}
