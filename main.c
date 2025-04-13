#include <float.h>
#include <math.h>
#include <raylib.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"
#include <assert.h>

#define WIDTH 1920
#define HEIGHT 1080

#define MAT_WITHIN(mat, row, col)                                              \
  (0 <= (col) && (col) < (mat).width && 0 <= (row) && (row) < (mat).height)
#define MAT_AT(mat, row, col, stride) (mat).data[(row) * stride + (col)]

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

typedef struct {
  float *data;
  int width;
  int height;
  int stride;
} Mat;

static float sobel_filter_at(Mat img, int cx, int cy) {
  float sx = 0.0f;
  float sy = 0.0f;

  for (int dy = -1; dy <= 1; dy++) {
    for (int dx = -1; dx <= 1; dx++) {
      int x = cx + dx;
      int y = cy + dy;
      float c = MAT_WITHIN(img, y, x) ? MAT_AT(img, y, x, img.stride) : 0.0;
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
      MAT_AT(gradient, cy, cx, img.stride) = sobel_filter_at(img, cx, cy);
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
    MAT_AT(dp, 0, x, dp.stride) = MAT_AT(gradient, 0, x, gradient.stride);
  }

  for (int y = 1; y < gradient.height; y++) {
    for (int cx = 0; cx < gradient.width; cx++) {
      // Compute minimal value moving down left, down or down right
      float m = FLT_MAX;
      for (int dx = -1; dx <= 1; dx++) {
        int x = cx + dx;
        float c = (0 <= x && x < gradient.width)
                      ? MAT_AT(dp, y - 1, x, dp.stride)
                      : FLT_MAX;
        if (c < m)
          m = c;
      }
      MAT_AT(dp, y, cx, dp.stride) =
          MAT_AT(gradient, y, cx, gradient.stride) + m;
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

static void compute_seam2(Mat dp, int *seam) {
  int y = dp.height - 1;
  seam[y] = 0;
  for (int x = 1; x < dp.width; ++x) {
    if (MAT_AT(dp, y, x, dp.stride) < MAT_AT(dp, y, seam[y], dp.stride)) {
      seam[y] = x;
    }
  }

  for (y = dp.height - 2; y >= 0; --y) {
    seam[y] = seam[y + 1];
    for (int dx = -1; dx <= 1; ++dx) {
      int x = seam[y + 1] + dx;
      if (0 <= x && x < dp.width &&
          MAT_AT(dp, y, x, dp.stride) < MAT_AT(dp, y, seam[y], dp.stride)) {
        seam[y] = x;
      }
    }
  }
}

static void compute_seam(Mat dp, int *seam) {
  int y = dp.height - 1;
  seam[y] = 0;

  // Get minimum value at the last row
  for (int x = 1; x < dp.width; x++) {
    if (MAT_AT(dp, y, x, dp.stride) < MAT_AT(dp, y, seam[y], dp.stride)) {
      seam[y] = x;
    }
  }

  for (y = dp.height - 2; y >= 0; y--) {
    seam[y] = seam[y + 1]; // previous value
    for (int dx = -1; dx <= 1; dx++) {
      int x = seam[y + 1] + dx;
      if ((0 <= x && x < dp.width) &&
          MAT_AT(dp, y, x, dp.stride) < MAT_AT(dp, y, seam[y], dp.stride)) {
        seam[y] = x;
      }
    }
  }
}

static void img_remove_column_at_row(Image img, int y, int x, int stride) {
  Color *data = img.data;
  Color *pixel_row = &data[y * stride];
  memmove(pixel_row + x, pixel_row + x + 1, (stride - x - 1) * sizeof(Color));
}

static void mat_remove_column_at_row(Mat mat, int row, int column) {
  float *pixel_row = &MAT_AT(mat, row, 0, mat.stride);
  memmove(pixel_row + column, pixel_row + column + 1,
          (mat.stride - column - 1) * sizeof(float));
}

static Mat mat_alloc(int w, int h) {
  Mat mat = {0};
  mat.width = w;
  mat.height = h;
  mat.data = calloc(mat.width * mat.height, sizeof(*mat.data));
  mat.stride = w;
  assert(mat.data != NULL);
  return mat;
}

static Mat image_luminance(Image img) {
  Mat mat = mat_alloc(img.width, img.height);
  for (int x = 0; x < img.width; x++) {
    for (int y = 0; y < img.height; y++) {
      Color c = ((Color *)img.data)[y * img.width + x];
      MAT_AT(mat, y, x, mat.width) = rgb_to_luminance(c);
    }
  }

  return mat;
}

#define ARENA_SIZE HEIGHT *WIDTH * sizeof(Color)
static uint8_t arena[ARENA_SIZE] = {0};

static inline void *arena_alloc(size_t nb, size_t size) {
  assert(nb * size <= ARENA_SIZE);
  return arena;
}

static inline void arena_free() { memset(arena, 0, ARENA_SIZE); }

static Image img_alloc(Image original, int stride) {
  Image mat = {0};
  mat.width = original.width;
  mat.height = original.height;
  mat.format = original.format;
  mat.mipmaps = original.mipmaps;
  Color *new_data = arena_alloc(mat.width * mat.height, sizeof(Color));
  mat.data = new_data;

  Color *original_data = original.data;
  for (int y = 0; y < mat.height; y++) {
    for (int x = 0; x < mat.width; x++) {
      new_data[y * mat.width + x] = original_data[y * stride + x];
    }
  }

  return mat;
}

static void draw_mat(Image mat) {
  for (int y = 0; y < mat.height; y++) {
    for (int x = 0; x < mat.width; x++) {
      Color *data = mat.data;
      Color c = data[y * mat.width + x];
      DrawRectangle(WIDTH / 2 - mat.width / 2 + x,
                    HEIGHT / 2 - mat.height / 2 + y, 1, 1, c);
    }
  }
}

static Image mat_to_img(Mat mat, int mipmaps) {
  Image img = {0};
  img.width = mat.width;
  img.height = mat.height;
  img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
  img.mipmaps = 1;
  Color *data = calloc(mat.width * mat.height, sizeof(Color));
  img.data = data;
  int i = 0;
  for (int y = 0; y < mat.height; y++) {
    for (int x = 0; x < mat.width; x++) {
      uint8_t alpha = MAT_AT(mat, y, x, mat.stride);
      data[y * mat.width + x] = (Color){255, 255, 255, alpha};
    }
  }

  return img;
}

typedef enum {
  STATE_START,
  STATE_LUMINANCE,
  STATE_GRADIENT,
  STATE_SEAM_REMOVAL,
} State;

static State state = STATE_START;

Image img;
char *filepath;
int seams_removed;
Mat luminance;
Mat gradient;
Mat dp;

bool init = false;
Image initial_luminance;
Image initial_gradient;

void set_state() {
  seams_removed = 0;
  img = LoadImage(filepath);
  ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);

  luminance = image_luminance(img);
  gradient = mat_alloc(img.width, img.height);
  sobel_filter(luminance, gradient);

  if (!init) {
    initial_luminance = mat_to_img(luminance, img.mipmaps);
    initial_gradient = mat_to_img(gradient, img.mipmaps);
    init = true;
  }

  dp = mat_alloc(img.width, img.height);
}

void reset_state() {
  UnloadImage(img);
  free(dp.data);
  free(luminance.data);
  free(gradient.data);

  set_state();
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <image>", argv[0]);
    return 0;
  }

  filepath = argv[1];
  set_state();

  InitWindow(WIDTH, HEIGHT, "Seam carving");
  int stride = img.width;
  int *seam = calloc(img.height, sizeof(*seam));
  int seams_to_remove = 1000;
  seams_removed = 0;

  Texture final_tex = {0};
  bool paused = true;

  int frame = 0;
  size_t rate = 128;
  bool show_seam = true;
  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_ONE)) {
      reset_state();
      state = STATE_START;
    } else if (IsKeyPressed(KEY_TWO)) {
      state = STATE_LUMINANCE;
    } else if (IsKeyPressed(KEY_THREE)) {
      state = STATE_GRADIENT;
    } else if (IsKeyPressed(KEY_FOUR)) {
      reset_state();
      state = STATE_SEAM_REMOVAL;
    } else if (IsKeyPressed(KEY_UP)) {
      if (rate >= 2) {
        rate /= 2;
      }
    } else if (IsKeyPressed(KEY_DOWN)) {
      if (rate <= 1024) {
        rate *= 2;
      }
    } else if (IsKeyPressed(KEY_SPACE)) {
      paused = !paused;
    }

    BeginDrawing();
    ClearBackground(BLACK);
    switch (state) {
    case STATE_START: {
      Texture tex = LoadTextureFromImage(img);
      DrawTexture(tex, WIDTH / 2 - img.width / 2, HEIGHT / 2 - img.height / 2,
                  WHITE);
      break;
    }
    case STATE_LUMINANCE:
      draw_mat(initial_luminance);
      break;
    case STATE_GRADIENT:
      draw_mat(initial_gradient);
      break;
    case STATE_SEAM_REMOVAL: {
      Color *data = img.data;
      frame += 1;
      if (seams_removed < seams_to_remove) {
        if (final_tex.id != 0) {
          UnloadTexture(final_tex);
        }

        if (show_seam || paused) {
          gradient_to_dp(gradient, dp);
          compute_seam(dp, seam);
          for (int y = 0; y < img.height; y++) {
            int cx = seam[y];
            data[(y)*stride + (cx)] = RED;
          }
          if (frame % rate == 0) {
            show_seam = false;
          }
        } else {
          for (int cy = 0; cy < img.height; ++cy) {
            int cx = seam[cy];
            img_remove_column_at_row(img, cy, cx, stride);
            mat_remove_column_at_row(luminance, cy, cx);
            mat_remove_column_at_row(gradient, cy, cx);
          }

          img.width -= 1;
          luminance.width -= 1;
          gradient.width -= 1;
          dp.width -= 1;
          seams_removed += 1;
          show_seam = true;
        }
        Image new = img_alloc(img, stride);
        final_tex = LoadTextureFromImage(new);
      }
      DrawTexture(final_tex, WIDTH / 2 - img.width / 2,
                  HEIGHT / 2 - img.height / 2, WHITE);

      break;
    }
    }
    EndDrawing();
  }

  return 0;
}
