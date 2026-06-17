#ifndef TACO_VERSION_H
#define TACO_VERSION_H

// This file contains version/config info, gathered at cmake config time.

#define TACO_BUILD_TYPE "Release"
#define TACO_BUILD_DATE "2026-06-16"

#define TACO_BUILD_COMPILER_ID      "AppleClang"
#define TACO_BUILD_COMPILER_VERSION "15.0.0.15000309"

#define TACO_VERSION_MAJOR ""
#define TACO_VERSION_MINOR ""
// if taco starts using a patch number, add  here
// if taco starts using a tweak number, add  here

// For non-git builds, this will be an empty string.
#define TACO_VERSION_GIT_SHORTHASH "be9deffe"

#define TACO_FEATURE_OPENMP 0
#define TACO_FEATURE_PYTHON 0
#define TACO_FEATURE_CUDA   0

#endif /* TACO_VERSION_H */
