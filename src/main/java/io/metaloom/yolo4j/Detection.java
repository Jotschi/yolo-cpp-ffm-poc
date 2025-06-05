package io.metaloom.yolo4j;

public record Detection(BoundingBox box, float conf, int classId) {

	public String label() {
		return YoloLib.toLabel(this);
	}

}
