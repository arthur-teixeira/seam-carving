#include <float.h>
#include <math.h>
#include <raylib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#include <assert.h>

// #define WIDTH 600
// #define HEIGHT 480

#define WIDTH 1920
#define HEIGHT 1080

#define MAT_WITHIN(mat, row, col)                                              \
  (0 <= (col) && (col) < (mat).width && 0 <= (row) && (row) < (mat).height)
#define MAT_AT(mat, row, col) (mat).data[(row) * (mat).width + (col)]

static float sobel_x[3][3] = {
    {-1, 0, 1},
    {-2, 0, 2},
    {-1, 0, 1},
};

static float sobel_y[3][3] = {
    {-1, -2, -1},
    {0, 0, 0},
    {1, 2, 1},
};

// TODO: Instead of using the raw int values of the pixels, convert to luminance
typedef struct {
  float *data;
  int width;
  int height;
} Mat;

static float sobel_filter_at(Mat img, int cx, int cy) {
  float sx = 0.0f;
  float sy = 0.0f;

  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int x = cx + dx;
      int y = cy + dy;
      float c = MAT_WITHIN(img, y, x) ? MAT_AT(img, y, x) : 0.0;
      sx += c * sobel_x[dy + 1][dx + 1];
      sy += c * sobel_y[dy + 1][dx + 1];
    }
  }

  return sqrtf(sx * sx + sy * sy);
}

static void sobel_filter(Mat img, Mat gradient) {
  assert(img.width == gradient.width);
  assert(img.height == gradient.height);

  for (int cy = 0; cy < img.height; cy++) {
    for (int cx = 0; cx < img.width; cx++) {
      MAT_AT(gradient, cy, cx) = sobel_filter_at(img, cx, cy);
    }
  }
}

// Human perception of brightness according to ITU-R BT.709
static float rgb_to_luminance(Color c) {
  return 0.299 * c.r + 0.587 * c.g + 0.114 * c.b;
}

/*

The optimal seam can be found using dynamic programming. The
first step is to traverse the image from the second row to the last row
and compute the cumulative minimum energy M for all possible
connected seams for each entry (i, j):

M(i, j) = e(i, j)+ min(M(i−1, j −1),M(i−1, j),M(i−1, j +1))

 */

static void gradient_to_dp(Mat gradient, Mat dp) {
  assert(dp.width == gradient.width);
  assert(dp.height == gradient.height);

  for (int x = 0; x < gradient.width; x++) {
    // First row is a given
    MAT_AT(dp, 0, x) = MAT_AT(gradient, 0, x);
  }

  for (int y = 1; y < gradient.height; y++) {
    for (int cx = 0; cx < gradient.width; cx++) {
      // Compute minimal value moving down left, down or down right
      float m = FLT_MAX;
      for (int dx = -1; dx <= 1; dx++) {
        int x = cx + dx;
        float c =
            (0 <= x && x < gradient.width) ? MAT_AT(dp, y - 1, x) : FLT_MAX;
        if (c < m) {
          m = c;
        }
      }
      MAT_AT(dp, y, cx) = MAT_AT(gradient, y, cx) + m;
    }
  }
}

/*
At the end of this process, the minimum value of the last row in
M will indicate the end of the minimal connected vertical seam.
Hence, in the second step we backtrack from this minimum entry on
M to find the path of the optimal seam (see Figure 1). The definition
of M for horizontal seams is similar
*/

static void compute_seam(Mat dp, int *seam) {
  int y = dp.height - 1;
  seam[y] = 0;

  // Get minimum value at the last row
  for (int x = 1; x < dp.width; x++) {
    if (MAT_AT(dp, y, x) < MAT_AT(dp, y, seam[y])) {
      seam[y] = x;
    }
  }

  for (y = dp.height - 2; y >= 0; y--) {
    seam[y] = seam[y + 1]; // previous value
    for (int dx = -1; dx <= 1; dx++) {
      int x = seam[y + 1] + dx;
      if ((x >= 0 && x < dp.width) &&
          MAT_AT(dp, y, x) < MAT_AT(dp, y, seam[y])) {
        seam[y] = x;
      }
    }
  }
}

static Mat mat_alloc(int w, int h) {
  Mat mat = {0};
  mat.width = w;
  mat.height = h;
  mat.data = calloc(mat.width * mat.height, sizeof(*mat.data));
  assert(mat.data != NULL);
  return mat;
}

static Mat image_luminance(Image img) {
  Mat mat = mat_alloc(img.width, img.height);
  for (int x = 0; x < img.width; x++) {
    for (int y = 0; y < img.height; y++) {
      Color c = ((Color *)img.data)[y * img.width + x];
      MAT_AT(mat, y, x) = rgb_to_luminance(c);
    }
  }

  return mat;
}

static bool is_in_seam(int *seam, size_t size, int y, int x) {
  for (int i = 0; i < size; i++) {
    int sy = i;
    int sx = seam[i];
    if (sy == y && sx == x) {
      return true;
    }
  }

  return false;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <image>", argv[0]);
    return 0;
  }

  char *filepath = argv[1];

  InitWindow(WIDTH, HEIGHT, "Seam carving");

  Image original = LoadImage(filepath);
  ImageFormat(&original, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);

  Mat luminance = image_luminance(original);
  Mat gradient = mat_alloc(original.width, original.height);
  sobel_filter(luminance, gradient);

  Mat dp = mat_alloc(original.width, original.height);
  gradient_to_dp(gradient, dp);

  int *seam = calloc(original.height, sizeof(*seam));
  compute_seam(dp, seam);

  Color *data = original.data;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);

    for (int x = 0; x < original.width; x++) {
      for (int y = 0; y < original.height; y++) {
        int idx = y * original.width + x;
        Color c = GetColor(0xFFFFFF);

        if (is_in_seam(seam, original.height, y, x)) {
          c = RED;
        } else {
          c = data[idx];
        }

        DrawRectangle(WIDTH / 2 - original.width / 2 + x,
                      HEIGHT / 2 - original.height / 2 + y, 1, 1, c);
      }
    }

    EndDrawing();
  }

  return 0;
}
