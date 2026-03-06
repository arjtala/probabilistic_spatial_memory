CC = clang
CFLAGS = -Wall -Wextra -Werror -std=c99 -O3 -march=native -flto -ffast-math -mtune=native -Iinclude -I.
LDFLAGS = -flto
BUILD_DIR = build

# Vendor paths
VENDOR = vendor/probabilistic_data_structures
VENDOR_INCLUDES = -I$(VENDOR)/lib -I$(VENDOR)/hyperloglog -I$(VENDOR)/bloom_filter

# Headers
VENDOR_HEADERS = $(wildcard $(VENDOR)/lib/*.h) $(wildcard $(VENDOR)/hyperloglog/*.h) $(wildcard $(VENDOR)/bloom_filter/*.h)
LOCAL_HEADERS = $(wildcard include/*.h)
HEADERS = $(LOCAL_HEADERS) $(VENDOR_HEADERS)

# Source files
SRCS = $(wildcard src/*.c)
VENDOR_SRCS = $(VENDOR)/lib/hash.c $(VENDOR)/lib/bitarray.c $(VENDOR)/lib/utilities.c $(VENDOR)/hyperloglog/hll.c $(VENDOR)/bloom_filter/bloom.c

# Object files
OBJ = $(patsubst src/%.c,$(BUILD_DIR)/%.o,$(SRCS))
VENDOR_OBJ = $(patsubst $(VENDOR)/%.c,$(BUILD_DIR)/vendor/%.o,$(VENDOR_SRCS))

# Library name
LIB = libpsm.a

# Test executable
TEST_EXEC = $(BUILD_DIR)/test_ring_buffer

# Default target
all: $(LIB)

# Build the static library
$(LIB): $(OBJ) $(VENDOR_OBJ)
	ar rcs $@ $^

# Build project object files
$(BUILD_DIR)/%.o: src/%.c $(HEADERS)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

# Build vendor object files
$(BUILD_DIR)/vendor/%.o: $(VENDOR)/%.c $(VENDOR_HEADERS)
	@mkdir -p $(dir $@)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) -c $< -o $@

# Test target - build and run tests
test: $(TEST_EXEC)
	./$(TEST_EXEC)

# Build test executable
$(TEST_EXEC): tests/test_ring_buffer.c $(HEADERS) $(OBJ) $(VENDOR_OBJ)
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $(VENDOR_INCLUDES) $(LDFLAGS) tests/test_ring_buffer.c $(OBJ) $(VENDOR_OBJ) -o $@ -lm

# Clean target
clean:
	rm -rf $(BUILD_DIR) $(LIB)

# Rebuild everything
rebuild: clean all

# Phony targets
.PHONY: all test clean rebuild show

# Show detected files
show:
	@echo "Headers: $(HEADERS)"
	@echo "Sources: $(SRCS)"
	@echo "Vendor sources: $(VENDOR_SRCS)"
	@echo "Objects: $(OBJ) $(VENDOR_OBJ)"
