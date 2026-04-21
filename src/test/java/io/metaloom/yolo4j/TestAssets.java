package io.metaloom.yolo4j;

import java.io.IOException;
import java.io.InputStream;
import java.net.URI;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.StandardCopyOption;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

public final class TestAssets {

	private static final String YOLOS_CPP_VERSION = "1.0.0";
	private static final String MODELS_TAG = "v1.0.0-onnx-tuned-models";
	private static final String DETECTION_MODELS_ASSET = "yolo-detection-models-tuned.zip";

	private TestAssets() {
	}

	public static Path yoloCppDir() {
		Path v1Dir = Paths.get("YOLOs-CPP-" + YOLOS_CPP_VERSION);
		if (Files.isDirectory(v1Dir)) {
			return v1Dir;
		}
		Path legacyDir = Paths.get("YOLOs-CPP");
		return legacyDir;
	}

	public static Path labelsPath(Path modelPath) {
		Path modelsDir = yoloCppDir().resolve("models");
		String filename = modelPath.getFileName().toString().toLowerCase();
		Path preferred = filename.contains("_voc") ? modelsDir.resolve("voc.names") : modelsDir.resolve("coco.names");
		if (Files.exists(preferred)) {
			return preferred;
		}
		Path fallback = preferred.getFileName().toString().equals("voc.names") ? modelsDir.resolve("coco.names") : modelsDir.resolve("voc.names");
		if (Files.exists(fallback)) {
			return fallback;
		}
		return preferred;
	}

	public static Path imagePath() {
		Path dogImage = yoloCppDir().resolve("data").resolve("dog.jpg");
		if (Files.exists(dogImage)) {
			return dogImage;
		}
		return Paths.get("YOLOs-CPP/data/kitchen.jpg");
	}

	public static Path ensureDetectionModel() throws IOException {
		Path modelsDir = yoloCppDir().resolve("models");
		Files.createDirectories(modelsDir);

		Path preferred = modelsDir.resolve("YOLOv11n_voc.onnx");
		if (Files.exists(preferred)) {
			return preferred;
		}

		Path existing = firstOnnxModel(modelsDir);
		if (existing != null) {
			return existing;
		}

		downloadAndExtractDetectionModels(modelsDir);

		if (Files.exists(preferred)) {
			return preferred;
		}

		existing = firstOnnxModel(modelsDir);
		if (existing != null) {
			return existing;
		}

		throw new IOException("No detection ONNX model found in " + modelsDir);
	}

	private static Path firstOnnxModel(Path modelsDir) throws IOException {
		try (Stream<Path> stream = Files.list(modelsDir)) {
			List<Path> models = stream
				.filter(path -> Files.isRegularFile(path) && path.getFileName().toString().toLowerCase().endsWith(".onnx"))
				.sorted(Comparator.comparing(path -> path.getFileName().toString()))
				.toList();
			return models.isEmpty() ? null : models.get(0);
		}
	}

	private static void downloadAndExtractDetectionModels(Path modelsDir) throws IOException {
		String url = "https://github.com/Geekgineer/YOLOs-CPP/releases/download/" + MODELS_TAG + "/" + DETECTION_MODELS_ASSET;
		Path zipPath = Files.createTempFile("yolo4j-detection-models", ".zip");
		zipPath.toFile().deleteOnExit();

		try (InputStream in = URI.create(url).toURL().openStream()) {
			Files.copy(in, zipPath, StandardCopyOption.REPLACE_EXISTING);
		}

		try (InputStream fis = Files.newInputStream(zipPath);
			ZipInputStream zis = new ZipInputStream(fis)) {
			ZipEntry entry;
			while ((entry = zis.getNextEntry()) != null) {
				if (entry.isDirectory()) {
					continue;
				}
				String fileName = Paths.get(entry.getName()).getFileName().toString();
				Path target = modelsDir.resolve(fileName);
				Files.copy(zis, target, StandardCopyOption.REPLACE_EXISTING);
			}
		}
	}
}