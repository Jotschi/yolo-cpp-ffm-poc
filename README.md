# YOLO4j

This library enables native access to YOLO object detection in Java.

Under the hood this library uses the Foreign Function and Memory API to hook into a custom library which uses YOLOs-CPP to run inference on OpenCV Mats which can be provided by Video4j.

![example](video.gif "Example")

## Limitations

Currently only AMD64 Linux is supported. Support for other platforms is not planned.

## Usage

```xml
<dependency>
  <groupId>io.metaloom.yolo4j</groupId>
  <artifactId>yolo4j</artifactId>
	<version>0.1.0-SNAPSHOT</version>
</dependency>
```

Image Example
```java
String imagePath = "YOLOs-CPP/data/kitchen.jpg";
boolean useGPU = true;

// Initialize video4j and YoloLib (Video4j is used to handle OpenCV Mat)
Video4j.init();
YoloLib.init("YOLOs-CPP-1.0.0/models/YOLOv11n_voc.onnx", "YOLOs-CPP-1.0.0/models/voc.names", useGPU);

// Load the image and invoke the detection
BufferedImage img = ImageUtils.load(new File(imagePath));
List<Detection> detections = YoloLib.detect(img, false);

// Print the detections
for (Detection detection : detections) {
	System.out.println(detection.label() + " = " + detection.conf() + " @ " + detection.box());
}
```


Video Example
```java
boolean useGPU = false;

// Initialize video4j and YoloLib (Video4j is used to handle OpenCV Mat)
Video4j.init();
YoloLib.init("YOLOs-CPP-1.0.0/models/YOLOv11n_voc.onnx", "YOLOs-CPP-1.0.0/models/voc.names", useGPU);
SimpleImageViewer viewer = new SimpleImageViewer();

// Open the video using Video4j
try (VideoFile video = VideoFile.open("src/test/resources/3769953-hd_1920_1080_25fps.mp4")) {

	// Process each frame
	VideoFrame frame;
	while ((frame = video.frame()) != null) {
		System.out.println(frame);
		CVUtils.resize(frame, 1024);
		// Run the detection on the mat reference
		List<Detection> detections = YoloLib.detect(frame.mat(), true);

		// Print the detections
		for (Detection detection : detections) {
			String label = detection.label();
			double confidence = detection.conf();
			BoundingBox box = detection.box();
			System.out.println("Frame[" + video.currentFrame() + "] " + label + " = " + confidence + " @ " + box);
		}

		viewer.show(frame.mat());
	}
}
```


## Build 

### Requirements:

- YOLOs-CPP v1.0.0 [Release](https://github.com/Geekgineer/YOLOs-CPP/releases/tag/v1.0.0)
- ONNX Runtime shared library (required by `libyolib`)
- OpenCV development package with CMake config (`OpenCVConfig.cmake`)
- JDK 23 or newer
- Maven
- GCC 13

### YOLOs-CPP Source And Models

YOLOs-CPP v1.0.0 no longer ships models in the repository. Download source and models separately from GitHub Releases.

1. Download YOLOs-CPP source release:

```bash
curl -L https://github.com/Geekgineer/YOLOs-CPP/archive/refs/tags/v1.0.0.tar.gz -o yolo-cpp-v1.0.0.tar.gz
tar -xzf yolo-cpp-v1.0.0.tar.gz
```

2. Download detection ONNX models from the tuned model release tag (`v1.0.0-onnx-tuned-models`):

```bash
curl -L https://github.com/Geekgineer/YOLOs-CPP/releases/download/v1.0.0-onnx-tuned-models/yolo-detection-models-tuned.zip -o yolo-detection-models-tuned.zip
unzip -o yolo-detection-models-tuned.zip -d YOLOs-CPP-1.0.0/models
```

The tuned model release also provides additional ONNX assets:

- `yolo-segmentation-models-tuned.zip`
- `yolo-pose-models-tuned.zip`
- `yolo-obb-models-tuned.zip`
- `yolo-classification-models-tuned.zip`

Reference:

- Source release: https://github.com/Geekgineer/YOLOs-CPP/releases/tag/v1.0.0
- ONNX tuned model assets: https://github.com/Geekgineer/YOLOs-CPP/releases/tag/v1.0.0-onnx-tuned-models

### ONNX Runtime

`libyolib` links against ONNX Runtime at runtime. On Linux, `libonnxruntime.so.1` must be available.

Yolo4j tries to load ONNX Runtime automatically during `YoloLib.init(...)` using this order:

1. System property `yolo4j.onnxruntime.lib` (directory containing `libonnxruntime.so.1`)
2. Dynamic discovery relative to the provided model path (`onnxruntime-*/lib` folders in parent directories)

If ONNX Runtime cannot be resolved, initialization fails with an `UnsatisfiedLinkError`.

Example override:

```bash
java -Dyolo4j.onnxruntime.lib=/absolute/path/to/onnxruntime/lib ...
```

### Building native code

```bash
cd yolib
./build.sh 1.20.1 1 1.0.0
```

### Running Tests

Tests use YOLOs-CPP v1.0.0 paths and will try to download the detection model asset automatically if it is missing.

```bash
mvn test
```

## Releasing

```bash
# Set release version and commit changes
mvn versions:set -DgenerateBackupPoms=false
git add pom.xml ; git commit -m "Prepare release"

# Invoke release
mvn clean deploy -Drelease
```

