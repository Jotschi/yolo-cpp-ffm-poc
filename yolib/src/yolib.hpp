#ifndef YOLOLIB_HPP
#define YOLOLIB_HPP

#include <vector>
#include <opencv2/highgui/highgui.hpp>

typedef struct
{
    int x;
    int y;
    int width;
    int height;
} BoundingBox;

typedef struct
{
    BoundingBox box;
    float conf;
    int classId;
} Detection;

typedef struct
{
    Detection *data;
    int count;
} DetectionArray;


#endif // YOLOLIB_HPP