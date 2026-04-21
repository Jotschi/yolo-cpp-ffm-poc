package io.metaloom.yolo4j;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assumptions.assumeTrue;

import java.awt.GraphicsEnvironment;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

import io.metaloom.opencv.core.Mat;
import io.metaloom.opencv.imgproc.Imgproc;
import io.metaloom.video4j.Video4j;
import io.metaloom.video4j.VideoFile;
import io.metaloom.video4j.VideoFrame;
import io.metaloom.video4j.impl.MatProvider;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.video4j.utils.ImageUtils;
import io.metaloom.video4j.utils.SimpleImageViewer;

public class YoloLibTest {

	private static Path imagePath = TestAssets.imagePath();
	private static Path labelsPath;
	private static Path modelPath;

	@BeforeAll
	public static void setup() {
		Video4j.init();
		try {
			modelPath = TestAssets.ensureDetectionModel();
			labelsPath = TestAssets.labelsPath(modelPath);
			YoloLib.init(modelPath.toString(), labelsPath.toString(), false);
		} catch (RuntimeException e) {
			if (e.getMessage() != null && e.getMessage().contains("already initialized")) {
				return;
			}
			throw e;
		} catch (Throwable t) {
			throw new RuntimeException("Failed to initialize YoloLib test assets", t);
		}
	}

	@Test
	public void testImage() throws Throwable {
		// System.setProperty("java.library.path", onnxLibPath);
		BufferedImage img = ImageUtils.load(new File(imagePath.toString()));
		Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
		CVUtils.bufferedImageToMat(img, imageMat);
		List<Detection> detections = YoloLib.detect(imageMat, true);
		assertNotNull(detections);
		assertTrue(!detections.isEmpty(), "Expected at least one detection with bundled test assets");
		if (!GraphicsEnvironment.isHeadless()) {
			ImageUtils.show(imageMat);
		}

		for (Detection detection : detections) {
			System.out.println(detection.label() + " conf: " + detection.conf());
		}

	}

	@Test
	public void testVideo() throws Throwable {
		assumeTrue(!GraphicsEnvironment.isHeadless(), "Skipping viewer-based test in headless environment.");
		SimpleImageViewer viewer = new SimpleImageViewer();

		try (VideoFile video = VideoFile.open("src/test/resources/3769953-hd_1920_1080_25fps.mp4")) {
			video.seekToFrameRatio(0.5);
			long start = System.currentTimeMillis();

			VideoFrame frame;
			while ((frame = video.frame()) != null) {
				Mat imageMat = frame.mat();
				YoloLib.detect(imageMat, true);
				viewer.show(imageMat);
			}
			long dur = System.currentTimeMillis() - start;
			System.out.println("Took " + dur);
			Thread.sleep(100);
		}

	}
}
