package io.metaloom.yolo4j;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.lang.foreign.AddressLayout;
import java.lang.foreign.Arena;
import java.lang.foreign.FunctionDescriptor;
import java.lang.foreign.Linker;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.SymbolLookup;
import java.lang.foreign.ValueLayout;
import java.lang.foreign.ValueLayout.OfInt;
import java.lang.foreign.ValueLayout.OfLong;
import java.lang.invoke.MethodHandle;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;
import java.util.stream.Stream;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import io.metaloom.opencv.core.Mat;
import io.metaloom.opencv.imgproc.Imgproc;
import io.metaloom.video4j.impl.MatProvider;
import io.metaloom.video4j.opencv.CVUtils;
import io.metaloom.yolo4j.layout.DetectionArrayMemoryLayout;
import io.metaloom.yolo4j.layout.DetectionMemoryLayout;

public class YoloLib {

	private static final Logger logger = LoggerFactory.getLogger(YoloLib.class);

	// private static String onnxLibFileName = "libonnxruntime.so.1.20.1";
	// private static String onnxLibName = "onnxruntime";
	// private static String onnxLibPath = "YOLOs-CPP/onnxruntime-linux-x64-1.20.1/lib/";
	// private static String onnxLib = "libonnxruntime.so.1.20.1";

	// private static String libPath = "yolib/build/libyolib.so.1.0.0";

	static final Linker linker = Linker.nativeLinker();
	static final Arena arena = Arena.ofAuto();

	static final AddressLayout ADDR = ValueLayout.ADDRESS.withOrder(ByteOrder.nativeOrder());
	static final OfInt JINT = ValueLayout.JAVA_INT.withOrder(ByteOrder.nativeOrder());
	static final OfLong JSIZE_T = ValueLayout.JAVA_LONG.withOrder(ByteOrder.nativeOrder()); // size_t on 64-bit Linux

	private static SymbolLookup yoloLibrary;

	private static boolean initialized = false;

	private static List<String> labels;

	private static void checkInitialized() {
		if (!initialized) {
			throw new RuntimeException("YoloLib not initialized");
		}
	}

	private static List<Detection> mapDetectionsArray(MemorySegment detectionArrayStruct) throws Throwable {

		MethodHandle freeDetectionHandler = linker.downcallHandle(
			yoloLibrary.find("free_detection").orElseThrow(),
			FunctionDescriptor.ofVoid(ValueLayout.ADDRESS));

		detectionArrayStruct = detectionArrayStruct.reinterpret(DetectionArrayMemoryLayout.size());

		MemorySegment dataPtr = detectionArrayStruct.get(ValueLayout.ADDRESS, 0);
		int detectionCount = detectionArrayStruct.get(ValueLayout.JAVA_INT, ValueLayout.ADDRESS.byteSize());
		dataPtr = dataPtr.reinterpret(DetectionMemoryLayout.size() * detectionCount);

		if (logger.isDebugEnabled()) {
			logger.debug("Received " + detectionCount + " detections");
			logger.debug("Got " + detectionArrayStruct + " from function.");
		}

		// Read BoundingBox elements
		List<Detection> detections = new ArrayList<>();
		long structSize = DetectionMemoryLayout.size();
		for (long i = 0; i < detectionCount; i++) {
			MemorySegment detectionMemory = dataPtr.asSlice(i * structSize, structSize);
			float conf = DetectionMemoryLayout.getConf(detectionMemory);
			int clazz = DetectionMemoryLayout.getClassId(detectionMemory);
			int x = detectionMemory.get(JINT, 0);
			int y = detectionMemory.get(JINT, JINT.byteSize());
			int width = detectionMemory.get(JINT, 2 * JINT.byteSize());
			int height = detectionMemory.get(JINT, 3 * JINT.byteSize());
			BoundingBox bbox = new BoundingBox(x, y, width, height);
			detections.add(new Detection(bbox, conf, clazz));
		}

		// Print results
		// detections.forEach(System.out::println);

		// Free the native memory
		freeDetectionHandler.invoke(detectionArrayStruct);
		return detections;

	}

	private static SymbolLookup loadLib(Path modelPath) {
		String os = System.getProperty("os.name", "generic")
			.toLowerCase(Locale.ENGLISH);
		String name = System.mapLibraryName("yolib");
		loadOnnxRuntime(os, modelPath);

		String libpath = "";
		if (os.contains("linux")) {
			libpath = "/native" + File.separator + "linux" + File.separator + name;
		} else if (os.contains("mac")) {
			libpath = "/native" + File.separator + "macosx" + File.separator + name;
		} else {
			throw new java.lang.UnsupportedOperationException(os + " is not supported. Try to recompile Jdlib on your machine and then use it.");
		}

		try (InputStream inputStream = YoloLib.class.getResourceAsStream(libpath)) {
			if (inputStream == null) {
				throw new RuntimeException("Could not find native resource " + libpath);
			}
			File fileOut = File.createTempFile(name, "");
			fileOut.deleteOnExit();
			try (OutputStream outputStream = new FileOutputStream(fileOut)) {
				byte[] buffer = new byte[1024];
				int read = -1;
				while ((read = inputStream.read(buffer)) != -1) {
					outputStream.write(buffer, 0, read);
				}
				System.load(fileOut.toString());
			}
			return SymbolLookup.loaderLookup();
			// yoloLibrary = SymbolLookup.libraryLookup(libPath, arena);
		} catch (Exception e) {
			throw new RuntimeException("Failed to load jdlib native library.", e);
		}
	}

	private static void loadOnnxRuntime(String os, Path modelPath) {
		String runtimeLibName = runtimeLibName(os);
		if (runtimeLibName == null) {
			return;
		}

		List<Path> attempted = new ArrayList<>();
		for (Path libDir : onnxRuntimeSearchPaths(modelPath)) {
			Path runtime = libDir.resolve(runtimeLibName);
			attempted.add(runtime.toAbsolutePath());
			if (!Files.exists(runtime)) {
				continue;
			}
			try {
				System.load(runtime.toAbsolutePath().toString());
				logger.debug("Loaded ONNX Runtime from {}", runtime.toAbsolutePath());
				return;
			} catch (UnsatisfiedLinkError e) {
				logger.warn("Failed to load ONNX Runtime from {}", runtime.toAbsolutePath(), e);
			}
		}
		logger.debug("Could not preload ONNX runtime. Checked: {}", attempted);
	}

	private static String runtimeLibName(String os) {
		if (os.contains("linux")) {
			return "libonnxruntime.so.1";
		}
		if (os.contains("mac")) {
			return "libonnxruntime.dylib";
		}
		return null;
	}

	private static List<Path> onnxRuntimeSearchPaths(Path modelPath) {
		String configuredPath = System.getProperty("yolo4j.onnxruntime.lib");
		Set<Path> paths = new LinkedHashSet<>();
		if (configuredPath != null && !configuredPath.isBlank()) {
			paths.add(Paths.get(configuredPath));
		}

		Path current = modelPath.toAbsolutePath().normalize();
		while (current != null) {
			if (current.getFileName() != null && current.getFileName().toString().startsWith("onnxruntime-")) {
				paths.add(current.resolve("lib"));
			}
			if (Files.isDirectory(current)) {
				try (Stream<Path> children = Files.list(current)) {
					children
						.filter(Files::isDirectory)
						.filter(path -> path.getFileName() != null && path.getFileName().toString().startsWith("onnxruntime-"))
						.map(path -> path.resolve("lib"))
						.forEach(paths::add);
				} catch (IOException e) {
					logger.debug("Skipping ONNX runtime scan for {}", current, e);
				}
			}
			current = current.getParent();
		}

		return new ArrayList<>(paths);
	}

	public static void init(String modelPath, String labelsPath, boolean useGPU) {
		if (initialized) {
			throw new RuntimeException("YoloLib already initialized");
		}
		Path mPath = Paths.get(modelPath);
		if (!Files.exists(mPath)) {
			throw new RuntimeException("Unable to locate model with path " + mPath);
		}

		Path lPath = Paths.get(labelsPath);
		if (!Files.exists(lPath)) {
			throw new RuntimeException("Unable to locate labels file with path " + lPath);
		}

		yoloLibrary = loadLib(mPath);

		try {
			labels = loadLabels(lPath);
		} catch (IOException e) {
			throw new RuntimeException("Failed to read labels from " + lPath);
		}

		MethodHandle initHandler = linker
			.downcallHandle(
				yoloLibrary.findOrThrow("initialize"),
				FunctionDescriptor.ofVoid(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));

		try (Arena arena = Arena.ofConfined()) {
			MemorySegment labelsPathMem = arena.allocateFrom(labelsPath);
			MemorySegment modelPathMem = arena.allocateFrom(modelPath);
			initHandler.invoke(labelsPathMem, modelPathMem, useGPU);
			initialized = true;
		} catch (Throwable t) {
			throw new RuntimeException("Failed to initialize YoloLib", t);
		}
	}

	private static List<String> loadLabels(Path path) throws IOException {
		return Files.readAllLines(path);
	}

	public static List<Detection> detect(Mat imageMat, boolean drawBoundingBoxes) {
		checkInitialized();

		MethodHandle detectHandler = linker
			.downcallHandle(
				yoloLibrary.findOrThrow("detect"),
				FunctionDescriptor.of(ValueLayout.ADDRESS, ValueLayout.ADDRESS, ValueLayout.JAVA_BOOLEAN));

		try {
			MemorySegment imageSeg = imageMat.nativePtr();
			MemorySegment detectionArrayStruct = (MemorySegment) detectHandler.invoke(imageSeg, drawBoundingBoxes);
			List<Detection> results = mapDetectionsArray(detectionArrayStruct);
			return results;
		} catch (Throwable t) {
			throw new RuntimeException("Failed to invoke detection", t);
		}

	}

	public static String toLabel(Detection detection) {
		int clazzId = detection.classId();
		if (clazzId < 1 || clazzId > labels.size()) {
			return null;
		}
		return labels.get(detection.classId());
	}

	public static List<Detection> detect(BufferedImage img, boolean drawBoundingBoxes) {
		Mat imageMat = MatProvider.mat(img, Imgproc.COLOR_BGRA2BGR565);
		CVUtils.bufferedImageToMat(img, imageMat);
		return detect(imageMat, drawBoundingBoxes);
	}
}
