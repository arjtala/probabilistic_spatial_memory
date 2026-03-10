CC = clang
H3_PREFIX = $(shell brew --prefix h3)
HDF5_PREFIX = $(shell brew --prefix hdf5)
GLFW_PREFIX = $(shell brew --prefix glfw)
FFMPEG_PREFIX = $(shell brew --prefix ffmpeg)
CURL_CFLAGS = $(shell curl-config --cflags)
CURL_LDFLAGS = $(shell curl-config --libs)
CFLAGS = -Wall -Wextra -Werror -std=c99 -O3 -march=native -flto -ffast-math -mtune=native -Iinclude -I. -I$(H3_PREFIX)/include -I$(HDF5_PREFIX)/include
LDFLAGS = -flto -L$(H3_PREFIX)/lib -lh3 -L$(HDF5_PREFIX)/lib -lhdf5
VIZ_CFLAGS = -I$(GLFW_PREFIX)/include -I$(FFMPEG_PREFIX)/include $(CURL_CFLAGS) -Ivendor -DGL_SILENCE_DEPRECATION
VIZ_LDFLAGS = -L$(GLFW_PREFIX)/lib -lglfw \
              -L$(FFMPEG_PREFIX)/lib -lavcodec -lavformat -lswscale -lavutil \
              $(CURL_LDFLAGS) \
              -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
BUILD_DIR = build

# Vendor paths
VENDOR = vendor/probabilistic_data_structures
VENDOR_INCLUDES = -I$(VENDOR)/lib -I$(VENDOR)/hyperloglog -I$(VENDOR)/bloom_filter

# Headers
VENDOR_HEADERS = $(wildcard $(VENDOR)/lib/*.h) $(wildcard $(VENDOR)/hyperloglog/*.h) $(wildcard $(VENDOR)/bloom_filter/*.h)
LOCAL_HEADERS = $(wildcard include/core/*.h) $(wildcard include/ingest/*.h) $(wildcard include/viz/*.h)
HEADERS = $(LOCAL_HEADERS) $(VENDOR_HEADERS)

# Source files
CORE_SRCS = $(wildcard src/core/*.c)
INGEST_SRCS = $(wildcard src/ingest/*.c)
SRCS = $(CORE_SRCS) $(INGEST_SRCS)
VENDOR_SRCS = $(VENDOR)/lib/hash.c $(VENDOR)/lib/bitarray.c $(VENDOR)/lib/utilities.c $(VENDOR)/hyperloglog/hll.c $(VENDOR)/bloom_filter/bloom.c

# Object files
CORE_OBJ = $(patsubst src/core/%.c,$(BUILD_DIR)/core/%.o,$(CORE_SRCS))
INGEST_OBJ = $(patsubst src/ingest/%.c,$(BUILD_DIR)/ingest/%.o,$(INGEST_SRCS))
OBJ = $(CORE_OBJ) $(INGEST_OBJ)
VENDOR_OBJ = $(patsubst $(VENDOR)/%.c,$(BUILD_DIR)/vendor/%.o,$(VENDOR_SRCS))

# Viz source/object files
VIZ_SRCS = src/viz/shader.c src/viz/video_decoder.c src/viz/hex_renderer.c src/viz/tile_map.c src/viz/gps_trace.c
VIZ_OBJ = $(patsubst src/viz/%.c,$(BUILD_DIR)/viz/%.o,$(VIZ_SRCS))
STB_OBJ = $(BUILD_DIR)/vendor/stb/stb_image_impl.o

# Output directory for binaries
TARGET_DIR = targets

# Library name
LIB = $(TARGET_DIR)/libpsm.a

# Main executable
BIN = $(TARGET_DIR)/psm
VIZ_BIN = $(TARGET_DIR)/psm-viz

# Test executables
TEST_RING_BUFFER = $(BUILD_DIR)/test_ring_buffer
TEST_TILE = $(BUILD_DIR)/test_tile
TEST_SPATIAL = $(BUILD_DIR)/test_spatial_memory

# Default target
all: $(LIB) $(BIN)

# Build the static library
$(LIB): $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(TARGET_DIR)
	ar rcs $@ $^

# Build main executable
$(BIN): src/main.c $(LIB)
	@mkdir -p $(TARGET_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) src/main.c $(LIB) -o $@ -lm

# Build project object files
$(BUILD_DIR)/core/%.o: src/core/%.c $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

$(BUILD_DIR)/ingest/%.o: src/ingest/%.c $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

# Build viz object files
$(BUILD_DIR)/viz/%.o: src/viz/%.c $(HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VIZ_CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

# Build stb_image object (suppress warnings from vendored header)
$(STB_OBJ): vendor/stb/stb_image_impl.c vendor/stb/stb_image.h
	@mkdir -p $(dir $@)
	$(CC) -std=c99 -O3 -Ivendor -w -c $< -o $@

# Build viz executable
viz: $(LIB) $(VIZ_OBJ) $(STB_OBJ)
	@mkdir -p $(TARGET_DIR)
	$(CC) $(CFLAGS) $(VIZ_CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) $(VIZ_LDFLAGS) src/viz/viz_main.c $(VIZ_OBJ) $(STB_OBJ) $(LIB) -o $(VIZ_BIN) -lm

# Build vendor object files
$(BUILD_DIR)/vendor/%.o: $(VENDOR)/%.c $(VENDOR_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

# Test targets
test: test-ring-buffer test-tile test-spatial

test-ring-buffer: $(TEST_RING_BUFFER)
	./$(TEST_RING_BUFFER)

test-tile: $(TEST_TILE)
	./$(TEST_TILE)

# Build test executables
$(TEST_RING_BUFFER): tests/test_ring_buffer.c $(HEADERS) $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_ring_buffer.c $(OBJ) $(VENDOR_OBJ) -o $@ -lm

$(TEST_TILE): tests/test_tile.c $(HEADERS) $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_tile.c $(OBJ) $(VENDOR_OBJ) -o $@ -lm

test-spatial: $(TEST_SPATIAL)
	./$(TEST_SPATIAL)

$(TEST_SPATIAL): tests/test_spatial_memory.c $(HEADERS) $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_spatial_memory.c $(OBJ) $(VENDOR_OBJ) -o $@ -lm

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(TARGET_DIR)

# Rebuild everything
rebuild: clean all

# Phony targets
.PHONY: all viz test test-ring-buffer test-tile test-spatial clean rebuild show run

# Show detected files
show:
	@echo "Headers: $(HEADERS)"
	@echo "Sources: $(SRCS)"
	@echo "Vendor sources: $(VENDOR_SRCS)"
	@echo "Objects: $(OBJ) $(VENDOR_OBJ)"
