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

.PHONY: all clean distclean test cjson