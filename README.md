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
  <version>0.0.1-SNAPSHOT</version>
</dependency>
```

Image Example
```java
String imagePath = "YOLOs-CPP/data/kitchen.jpg";
boolean useGPU = true;

// Initialize video4j and YoloLib (Video4j is used to handle OpenCV Mat)
Video4j.init();
YoloLib.init("YOLOs-CPP/models/yolo8n.onnx", "YOLOs-CPP/models/coco.names", useGPU);

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
YoloLib.init("YOLOs-CPP/models/yolo8n.onnx", "YOLOs-CPP/models/coco.names", useGPU);
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

- YOLO-CPP [Using YOLO-CPP from C++](https://github.com/Geekgineer/YOLOs-CPP)
- JDK 23 or newer
- Maven
- GCC 13

### Building native code

```bash
git clone git@github.com:Geekgineer/YOLOs-CPP.git (Head Rev: 363930885855b0441ba672d5ead7c6363cc34edb)
cd yolib
./build.sh 1.20.1 1
```

## Releasing

```bash
# Set release version and commit changes
mvn versions:set -DgenerateBackupPoms=false
git add pom.xml ; git commit -m "Prepare release"

# Invoke release
mvn clean deploy -Drelease
```

