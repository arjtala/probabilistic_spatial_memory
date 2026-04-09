CC = clang
UNAME_S := $(shell uname -s)
BREW ?= brew
PKG_CONFIG ?= pkg-config
comma := ,

pkg_prefix = $(strip $(shell $(PKG_CONFIG) --silence-errors --variable=prefix $(1) 2>/dev/null))
brew_prefix = $(strip $(shell $(BREW) --prefix $(1) 2>/dev/null))
rpath_flag = $(if $(1),-Wl$(comma)-rpath$(comma)$(1)/lib)

H3_PREFIX ?= $(or $(call pkg_prefix,h3),$(call pkg_prefix,libh3),$(call brew_prefix,h3))
HDF5_PREFIX ?= $(or $(call pkg_prefix,hdf5),$(call pkg_prefix,hdf5-serial),$(call brew_prefix,hdf5))
GLFW_PREFIX ?= $(or $(call pkg_prefix,glfw3),$(call brew_prefix,glfw))
FFMPEG_PREFIX ?= $(or $(call pkg_prefix,libavcodec),$(call brew_prefix,ffmpeg))
CURL_PREFIX ?= $(shell curl-config --prefix 2>/dev/null)
CURL_CFLAGS = $(shell curl-config --cflags 2>/dev/null)
CURL_LDFLAGS = $(shell curl-config --libs 2>/dev/null)

H3_CFLAGS = $(if $(H3_PREFIX),-I$(H3_PREFIX)/include)
H3_LDFLAGS = $(if $(H3_PREFIX),-L$(H3_PREFIX)/lib) -lh3
HDF5_CFLAGS = $(if $(HDF5_PREFIX),-I$(HDF5_PREFIX)/include)
HDF5_LDFLAGS = $(if $(HDF5_PREFIX),-L$(HDF5_PREFIX)/lib) -lhdf5
GLFW_CFLAGS = $(if $(GLFW_PREFIX),-I$(GLFW_PREFIX)/include)
GLFW_LDFLAGS = $(if $(GLFW_PREFIX),-L$(GLFW_PREFIX)/lib) -lglfw
FFMPEG_CFLAGS = $(if $(FFMPEG_PREFIX),-I$(FFMPEG_PREFIX)/include)
FFMPEG_LDFLAGS = $(if $(FFMPEG_PREFIX),-L$(FFMPEG_PREFIX)/lib) -lavcodec -lavformat -lswscale -lavutil

ifeq ($(UNAME_S),Darwin)
OPENGL_LDFLAGS = -framework OpenGL -framework Cocoa -framework IOKit -framework CoreVideo
TEST_OPENGL_LDFLAGS = -framework OpenGL
else ifeq ($(OS),Windows_NT)
OPENGL_LDFLAGS = -lopengl32 -lgdi32
TEST_OPENGL_LDFLAGS = -lopengl32
else
OPENGL_LDFLAGS = -lGL
TEST_OPENGL_LDFLAGS = -lGL
endif

ifeq ($(OS),Windows_NT)
PTHREAD_FLAGS =
else
PTHREAD_FLAGS = -pthread
endif

CFLAGS = -Wall -Wextra -Werror -std=c99 -O3 -march=native -flto -ffast-math -mtune=native -Iinclude -I. $(H3_CFLAGS) $(HDF5_CFLAGS)
LDFLAGS = -flto $(H3_LDFLAGS) $(HDF5_LDFLAGS)
VIZ_CFLAGS = $(GLFW_CFLAGS) $(FFMPEG_CFLAGS) $(CURL_CFLAGS) $(PTHREAD_FLAGS) -Ivendor -DGL_SILENCE_DEPRECATION
VIZ_RPATHS = $(call rpath_flag,$(H3_PREFIX)) \
             $(call rpath_flag,$(HDF5_PREFIX)) \
             $(call rpath_flag,$(GLFW_PREFIX)) \
             $(call rpath_flag,$(FFMPEG_PREFIX)) \
             $(call rpath_flag,$(CURL_PREFIX))
VIZ_LDFLAGS = $(PTHREAD_FLAGS) \
              $(GLFW_LDFLAGS) \
              $(FFMPEG_LDFLAGS) \
              $(CURL_LDFLAGS) \
              $(OPENGL_LDFLAGS)
BUILD_DIR = build
TOOLCHAIN_INFO = $(BUILD_DIR)/.toolchain-info

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
VIZ_SRCS = src/viz/shader.c src/viz/video_decoder.c src/viz/video_quad.c src/viz/progress_bar.c src/viz/attention_overlay.c src/viz/hex_renderer.c src/viz/tile_map.c src/viz/gps_trace.c src/viz/imu_processor.c src/viz/jepa_cache.c src/viz/viz_math.c src/viz/viz_config.c
VIZ_OBJ = $(patsubst src/viz/%.c,$(BUILD_DIR)/viz/%.o,$(VIZ_SRCS))
STB_OBJ = $(BUILD_DIR)/vendor/stb/stb_image_impl.o

# Output directory for binaries
TARGET_DIR = targets

# Library name
LIB = $(TARGET_DIR)/libpsm.a

# Main executable
BIN = $(TARGET_DIR)/psm
VIZ_BIN = $(TARGET_DIR)/psm-viz
BENCH_SPATIAL = $(TARGET_DIR)/benchmark_spatial_memory
BENCH_TILE_DECODE = $(TARGET_DIR)/benchmark_tile_decode

# Test executables
TEST_RING_BUFFER = $(BUILD_DIR)/test_ring_buffer
TEST_TILE = $(BUILD_DIR)/test_tile
TEST_TILE_TABLE = $(BUILD_DIR)/test_tile_table
TEST_SPATIAL = $(BUILD_DIR)/test_spatial_memory
TEST_INGEST = $(BUILD_DIR)/test_ingest
TEST_JEPA_CACHE = $(BUILD_DIR)/test_jepa_cache
TEST_VIZ_MATH = $(BUILD_DIR)/test_viz_math
TEST_VIZ_CONFIG = $(BUILD_DIR)/test_viz_config
TEST_GPS_TRACE = $(BUILD_DIR)/test_gps_trace

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

# Track toolchain/compiler changes so stale LLVM bitcode objects rebuild
# after SDK or compiler upgrades instead of surfacing linker warnings.
$(TOOLCHAIN_INFO): FORCE
	@mkdir -p $(BUILD_DIR)
	@current="$$( \
		printf 'CC=%s\n' '$(CC)'; \
		$(CC) --version 2>/dev/null | head -n 1; \
		printf 'SDK=%s\n' "$$(if command -v xcrun >/dev/null 2>&1; then xcrun --show-sdk-version 2>/dev/null; else printf unknown; fi)"; \
		printf 'ARCH=%s\n' "$$(uname -m 2>/dev/null || printf unknown)"; \
		printf 'CFLAGS=%s\n' '$(CFLAGS)'; \
		printf 'VIZ_CFLAGS=%s\n' '$(VIZ_CFLAGS)'; \
		printf 'LDFLAGS=%s\n' '$(LDFLAGS)'; \
		printf 'VIZ_LDFLAGS=%s\n' '$(VIZ_LDFLAGS)'; \
	)"; \
	if [ ! -f $@ ] || [ "$$(cat $@)" != "$$current" ]; then \
		printf '%s\n' "$$current" > $@; \
	fi

# Build project object files
$(BUILD_DIR)/core/%.o: src/core/%.c $(HEADERS) $(TOOLCHAIN_INFO)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

$(BUILD_DIR)/ingest/%.o: src/ingest/%.c $(HEADERS) $(TOOLCHAIN_INFO)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

# Build viz object files
$(BUILD_DIR)/viz/%.o: src/viz/%.c $(HEADERS) $(TOOLCHAIN_INFO)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VIZ_CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

# Build stb_image object (suppress warnings from vendored header)
$(STB_OBJ): vendor/stb/stb_image_impl.c vendor/stb/stb_image.h $(TOOLCHAIN_INFO)
	@mkdir -p $(dir $@)
	$(CC) -std=c99 -O3 -Ivendor -w -c $< -o $@

# Build viz executable
viz: $(LIB) $(VIZ_OBJ) $(STB_OBJ)
	@mkdir -p $(TARGET_DIR)
	$(CC) $(CFLAGS) $(VIZ_CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) $(VIZ_RPATHS) $(VIZ_LDFLAGS) src/viz/viz_main.c $(VIZ_OBJ) $(STB_OBJ) $(LIB) -o $(VIZ_BIN) -lm

bench-spatial-memory: $(BENCH_SPATIAL)
	./$(BENCH_SPATIAL)

bench-tile-decode: $(BENCH_TILE_DECODE)
	./$(BENCH_TILE_DECODE)

$(BENCH_SPATIAL): benchmarks/benchmark_spatial_memory.c $(LIB)
	@mkdir -p $(TARGET_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) benchmarks/benchmark_spatial_memory.c $(LIB) -o $@ -lm

$(BENCH_TILE_DECODE): benchmarks/benchmark_tile_decode.c $(STB_OBJ)
	@mkdir -p $(TARGET_DIR)
	$(CC) $(CFLAGS) $(PTHREAD_FLAGS) -Ivendor benchmarks/benchmark_tile_decode.c $(STB_OBJ) -o $@ -lz -lm

# Build vendor object files
$(BUILD_DIR)/vendor/%.o: $(VENDOR)/%.c $(VENDOR_HEADERS) $(TOOLCHAIN_INFO)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

# Test targets
test: test-ring-buffer test-tile test-tile-table test-spatial test-ingest test-jepa-cache test-viz-math test-viz-config test-gps-trace

test-ring-buffer: $(TEST_RING_BUFFER)
	./$(TEST_RING_BUFFER)

test-tile: $(TEST_TILE)
	./$(TEST_TILE)

test-tile-table: $(TEST_TILE_TABLE)
	./$(TEST_TILE_TABLE)

# Build test executables
$(TEST_RING_BUFFER): tests/test_ring_buffer.c $(HEADERS) $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_ring_buffer.c $(OBJ) $(VENDOR_OBJ) -o $@ -lm

$(TEST_TILE): tests/test_tile.c $(HEADERS) $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_tile.c $(OBJ) $(VENDOR_OBJ) -o $@ -lm

$(TEST_TILE_TABLE): tests/test_tile_table.c $(HEADERS) $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_tile_table.c $(OBJ) $(VENDOR_OBJ) -o $@ -lm

test-spatial: $(TEST_SPATIAL)
	./$(TEST_SPATIAL)

$(TEST_SPATIAL): tests/test_spatial_memory.c $(HEADERS) $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_spatial_memory.c $(OBJ) $(VENDOR_OBJ) -o $@ -lm

test-ingest: $(TEST_INGEST)
	./$(TEST_INGEST)

$(TEST_INGEST): tests/test_ingest.c $(HEADERS) $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_ingest.c $(OBJ) $(VENDOR_OBJ) -o $@ -lm

test-jepa-cache: $(TEST_JEPA_CACHE)
	./$(TEST_JEPA_CACHE)

$(TEST_JEPA_CACHE): tests/test_jepa_cache.c src/viz/jepa_cache.c include/viz/jepa_cache.h include/ingest/ingest.h build/vendor/lib/utilities.o
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_jepa_cache.c src/viz/jepa_cache.c build/vendor/lib/utilities.o -o $@ -lm

test-viz-math: $(TEST_VIZ_MATH)
	./$(TEST_VIZ_MATH)

$(TEST_VIZ_MATH): tests/test_viz_math.c src/viz/viz_math.c include/viz/viz_math.h include/viz/imu_processor.h build/vendor/lib/utilities.o
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) tests/test_viz_math.c src/viz/viz_math.c build/vendor/lib/utilities.o -o $@ -lm

test-viz-config: $(TEST_VIZ_CONFIG)
	./$(TEST_VIZ_CONFIG)

$(TEST_VIZ_CONFIG): tests/test_viz_config.c src/viz/viz_config.c include/viz/viz_config.h build/vendor/lib/utilities.o
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) tests/test_viz_config.c src/viz/viz_config.c build/vendor/lib/utilities.o -o $@

test-gps-trace: $(TEST_GPS_TRACE)
	./$(TEST_GPS_TRACE)

$(TEST_GPS_TRACE): tests/test_gps_trace.c src/viz/gps_trace.c src/viz/viz_math.c include/viz/gps_trace.h include/viz/viz_math.h include/viz/imu_processor.h build/vendor/lib/utilities.o $(TOOLCHAIN_INFO)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VIZ_CFLAGS) $(VENDOR_INCLUDES) tests/test_gps_trace.c src/viz/gps_trace.c src/viz/viz_math.c build/vendor/lib/utilities.o -o $@ $(TEST_OPENGL_LDFLAGS) -lm

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(TARGET_DIR)

# Rebuild everything
rebuild: clean all

# Phony targets
.PHONY: all viz bench-spatial-memory bench-tile-decode test test-ring-buffer test-tile test-tile-table test-spatial test-ingest test-jepa-cache test-viz-math test-viz-config test-gps-trace clean rebuild show run FORCE

# Show detected files
show:
	@echo "Headers: $(HEADERS)"
	@echo "Sources: $(SRCS)"
	@echo "Vendor sources: $(VENDOR_SRCS)"
	@echo "Objects: $(OBJ) $(VENDOR_OBJ)"
