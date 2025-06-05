#include <string>
#include "det/YOLO12.hpp"
#include "yolib.hpp"
#include <string.h>

// For Test code
#include <iostream>
#include <string>
#include <memory>

static std::unique_ptr<YOLO12Detector> globalDetector;

static bool initialized = false;

extern "C" void test_str(const char *name, const char *name2)
{
    printf("Test %s - %s\n", name, name2);
    fflush(stdout);
}

extern "C" void initialize(const char *labelsPath, const char *modelPath, bool useGPU)
{
    if (!initialized)
    {
        printf("Initializing YoloLib using model %s - labels %s - Using GPU: %s\n", modelPath, labelsPath, useGPU ? "true" : "false");
        fflush(stdout);
        globalDetector = std::make_unique<YOLO12Detector>(modelPath, labelsPath, useGPU);
        initialized = true;
    }
}

extern "C" void free_detection(DetectionArray *arr)
{
    if (arr)
    {
        delete[] arr->data;
        delete arr;
    }
}

extern "C" DetectionArray *detect_test()
{
    std::vector<Detection> tempVec;

    for (int i = 0; i < 4; i++)
    {
        BoundingBox bbox = {42 + i, 41, 40, 39 - i};
        Detection *d = new Detection;
        d->box = bbox;
        d->conf = 2.0f + i;
        d->classId = 11 + i;
        tempVec.push_back(*d);
    }

    DetectionArray *result = new DetectionArray;
    result->count = static_cast<int>(tempVec.size());
    result->data = new Detection[result->count];
    std::copy(tempVec.begin(), tempVec.end(), result->data);

    return result;
}

extern "C" DetectionArray *detect(cv::Mat *imagePtr, bool drawBoundingBox)
{

    YOLO12Detector *detector = globalDetector.get();
    cv::Mat image = *imagePtr;

    // Detect objects in the image and measure execution time
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<Detection> results = detector->detect(image);
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now() - start);
    printf("Detection of %i completed in: %i ms\n", results.size(), duration.count());
    fflush(stdout);
    DetectionArray *result = new DetectionArray;
    if (results.empty())
    {
        result->count = 0;
        result->data = new Detection[0];
        return result;
    }
    else
    {
        if (drawBoundingBox)
        {
            detector->drawBoundingBox(image, results);
        }

        result->count = static_cast<int>(results.size());
        result->data = new Detection[result->count];
        std::copy(results.begin(), results.end(), result->data);
        return result;
    }
}
