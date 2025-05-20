import argparse
import pathlib
import time
import uuid

import cv2
from insightface.app import FaceAnalysis
from insightface.utils.face_align import norm_crop

def main():
    ap = argparse.ArgumentParser(
        description="Capture and save face crops for the attendance dataset"
    )
    ap.add_argument("-n", "--name", required=True,
                    help="Your unique label (e.g., alice, bob_smith)")
    ap.add_argument("-o", "--out_dir", default="dataset_raw",
                    help="Root output folder (default: dataset_raw/)")
    ap.add_argument("-c", "--count", type=int, default=30,
                    help="Number of images to collect (default: 30)")
    args = ap.parse_args()

    # ------------ Paths ------------
    person_dir = pathlib.Path(args.out_dir) / args.name
    person_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Saving to: {person_dir.resolve()}")

    # ------------ Face detector / aligner ------------
    face_app = FaceAnalysis(name="buffalo_l", providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(0)
    saved = 0
    print("[INFO] Press SPACE to capture when your face is framed well.")
    print("       Press ESC to quit early.")

    while cap.isOpened() and saved < args.count:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame grab failed; retrying…")
            time.sleep(0.1)
            continue

        # Overlay saved count
        disp = frame.copy()
        cv2.putText(disp, f"Saved: {saved}/{args.count}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 0), 2)
        cv2.imshow("Capture - press SPACE", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:          # ESC -> quit
            break
        elif key == 32:        # SPACE -> capture
            faces = face_app.get(frame)
            if not faces:
                print("   ❌  No face detected – try again.")
                continue

            # Pick the largest detected face
            face = max(faces, key=lambda f: f.bbox[2] * f.bbox[3])

            # Align using norm_crop with a supported size (112×112)
            aligned = norm_crop(frame, face.kps, 112)

            # Convert to BGR for saving
            aligned_bgr = cv2.cvtColor(aligned, cv2.COLOR_RGB2BGR)

            # Save and increment
            fname = person_dir / f"{uuid.uuid4().hex}.jpg"
            cv2.imwrite(str(fname), aligned_bgr)
            saved += 1
            print(f"   ✅  Saved {fname.name}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Collected {saved} images for {args.name}")

if __name__ == "__main__":
    main()
