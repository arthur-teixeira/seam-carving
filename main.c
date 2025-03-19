#include <math.h>
#include <raylib.h>
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

static Mat mat_alloc(int w, int h) {
  Mat mat = {0};
  mat.width = w;
  mat.height = h;
  mat.data = malloc(mat.width * mat.height * sizeof(*mat.data));
  assert(mat.data != NULL);
  return mat;
}

static Mat image_to_matrix(Image img) {
  Mat mat = mat_alloc(img.width, img.height);
  for (int x = 0; x < img.width; x++) {
    for (int y = 0; y < img.height; y++) {
      Color c = ((Color *)img.data)[y * img.width + x];
      MAT_AT(mat, y, x) = rgb_to_luminance(c);
    }
  }

  return mat;
}

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <image>", argv[0]);
    return 0;
  }

  char *filepath = argv[1];

  InitWindow(WIDTH, HEIGHT, "Seam carving");

  Image img = LoadImage(filepath);
  ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
  Color *points = (Color *)img.data;

  Mat mat = image_to_matrix(img);
  Mat luminance = mat_alloc(img.width, img.height);

  float ww = img.width;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);
    GuiSliderBar((Rectangle){0, 0, WIDTH, 10}, "", TextFormat("%.2f", ww), &ww,
                 (float)img.width / 2, img.width * 1.5f);

    for (int x = 0; x < mat.width; x++) {
      for (int y = 0; y < mat.height; y++) {
        Color c = GetColor(0xFFFFFF);
        c.a = mat.data[y * mat.width + x];

        DrawRectangle(WIDTH / 2 - mat.width / 2 + x,
                      HEIGHT / 2 - mat.height / 2 + y, 1, 1, c);
      }
    }

    EndDrawing();
  }

  return 0;
}
