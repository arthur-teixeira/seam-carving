#include <raylib.h>
#include <stdio.h>
#define RAYGUI_IMPLEMENTATION
#include "raygui.h"

#define WIDTH 600
#define HEIGHT 480

#define WIDTH 1920
#define HEIGHT 1080

int main(int argc, char **argv) {
  if (argc < 2) {
    printf("Usage: %s <image>", argv[0]);
    return 0;
  }

  char *filepath = argv[1];

  InitWindow(WIDTH, HEIGHT, "Seam carving");

  Image img = LoadImage(filepath);
  ImageFormat(&img, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8);
  Color *points = img.data;
  Image resized = img;
  float ww = img.width;

  while (!WindowShouldClose()) {
    BeginDrawing();
    ClearBackground(BLACK);
    GuiSliderBar((Rectangle){0, 0, WIDTH, 10}, "", TextFormat("%.2f", ww), &ww,
                 (float)img.width / 2, img.width * 2);

    for (int x = 0; x < resized.width; x++) {
      for (int y = 0; y < resized.height; y++) {
        DrawRectangle(WIDTH / 2 - resized.width / 2 + x,
                      HEIGHT / 2 - resized.height / 2 + y, 1, 1,
                      points[y * resized.width + x]);
      }
    }

    EndDrawing();
  }

  return 0;
}
