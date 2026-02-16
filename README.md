# llama4aj

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## About

A Direct Android & Java Build For The [mybigday](https://github.com/mybigday) Serverless Server-like Chat Completion Implementation

Allows For Fast Inference & Intuitive App Building - Checkout The Example Apps & Try Building Them Yourself!

## Building

### Bootstrap (Required for first time or updates)

To synchronize the C++ sources and prepare the build environment:

```bash
./gradlew bootstrap
```

This will:
- Update `llama.rn` and `llama.cpp` submodules.
- Copy necessary source files to `cpp/`.
- Apply patches and renaming logic.

### Android

## Getting Started

```
./gradlew :examples:android-app:build
# Or For The More Simple Desktop App
./gradlew :examples:desktop-app:build
```

***COMING TO MAVEN SOON***

***SCALA VERSION COMING SOON***

## TODO

Proper bootstrap system

Proper syncing system

Add Desktop Example

Tests

Docs

Videos

Cleanup

[mybigday]:(https://github.com/mybigday)
