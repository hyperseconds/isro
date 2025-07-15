# Makefile for TITANUS CME Prediction Engine

CC = gcc
CFLAGS = -Wall -Wextra -O2 -std=c99 -g
LIBS = -lm

# Target executable
TARGET = titanus_predictor

# Source files
SOURCES = main.c feature_extractor.c

# Object files
OBJECTS = $(SOURCES:.c=.o)

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(OBJECTS) cjson
	$(CC) $(OBJECTS) cJSON.o -o $(TARGET) $(LIBS)

# Compile source files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Download and compile cJSON if not present
cjson:
	@if [ ! -f cJSON.c ]; then \
		echo "Downloading cJSON library..."; \
		curl -O https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.c; \
		curl -O https://raw.githubusercontent.com/DaveGamble/cJSON/master/cJSON.h; \
	fi
	$(CC) $(CFLAGS) -c cJSON.c -o cJSON.o

# Clean build artifacts
clean:
	rm -f $(OBJECTS) cJSON.o $(TARGET)

# Clean everything including downloaded files
distclean: clean
	rm -f cJSON.c cJSON.h

# Install dependencies (for systems with package managers)
deps:
	@echo "Installing dependencies..."
	@if command -v apt-get > /dev/null; then \
		sudo apt-get update && sudo apt-get install -y build-essential curl; \
	elif command -v yum > /dev/null; then \
		sudo yum install -y gcc make curl; \
	elif command -v brew > /dev/null; then \
		brew install gcc make curl; \
	else \
		echo "Please install gcc, make, and curl manually"; \
	fi

# Debug build
debug: CFLAGS += -DDEBUG -g3
debug: $(TARGET)

# Release build
release: CFLAGS += -DNDEBUG -O3
release: $(TARGET)

# Test compilation
test: $(TARGET)
	@echo "Testing compilation..."
	@if [ -f $(TARGET) ]; then \
		echo "✓ Compilation successful"; \
		echo "✓ Executable created: $(TARGET)"; \
	else \
		echo "✗ Compilation failed"; \
		exit 1; \
	fi

# Help target
help:
	@echo "TITANUS CME Prediction Engine Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all      - Build the prediction engine (default)"
	@echo "  clean    - Remove build artifacts"
	@echo "  distclean- Remove all generated files"
	@echo "  deps     - Install system dependencies"
	@echo "  debug    - Build with debug symbols"
	@echo "  release  - Build optimized release version"
	@echo "  test     - Test compilation"
	@echo "  help     - Show this help message"

.PHONY: all clean distclean deps debug release test help cjson
