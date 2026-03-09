import cv2
import numpy as np
import time


class RealTimeEnhancement:

    def __init__(self, target_fps=30):
        self.target_fps = target_fps
        self.history_buffer = []
        self.max_history = 5   # memory limitation

    # =====================================================
    # FRAME ENHANCEMENT
    # =====================================================
    def enhance_frame(self, frame, enhancement_type='adaptive'):
        """
        Enhance single frame with real-time constraints
        """

        start_time = time.time()

        # ---- Convert to grayscale ----
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # =================================================
        # Enhancement methods
        # =================================================
        if enhancement_type == 'adaptive':
            clahe = cv2.createCLAHE(
                clipLimit=2.0,
                tileGridSize=(8, 8)
            )
            enhanced = clahe.apply(gray)

        elif enhancement_type == 'histogram':
            enhanced = cv2.equalizeHist(gray)

        else:
            enhanced = gray

        # =================================================
        # TEMPORAL CONSISTENCY
        # smoothing antar frame
        # =================================================
        self.history_buffer.append(enhanced)

        if len(self.history_buffer) > self.max_history:
            self.history_buffer.pop(0)

        enhanced = np.mean(
            self.history_buffer,
            axis=0
        ).astype(np.uint8)

        # =================================================
        # Convert back to BGR
        # =================================================
        enhanced = cv2.cvtColor(
            enhanced,
            cv2.COLOR_GRAY2BGR
        )

        # =================================================
        # FPS CONTROL (computational constraint)
        # =================================================
        elapsed = time.time() - start_time
        delay = max(1 / self.target_fps - elapsed, 0)
        time.sleep(delay)

        return enhanced


# =====================================================
# REALTIME VIDEO STREAM
# =====================================================
def main():

    cap = cv2.VideoCapture(0)  # webcam
    enhancer = RealTimeEnhancement(target_fps=30)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        enhanced_frame = enhancer.enhance_frame(
            frame,
            enhancement_type='adaptive'
        )

        cv2.imshow("Original", frame)
        cv2.imshow("Enhanced", enhanced_frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()