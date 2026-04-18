"""Tests for input sources."""

from unittest.mock import MagicMock, patch

from src.input.image import ImageArraySource


class TestImageArraySource:
    def test_read_single_packet(self, sample_frame):
        source = ImageArraySource(sample_frame, name="uploaded.png")
        source.open()
        packet = source.read()
        assert packet is not None
        assert packet.source_id == "uploaded.png"
        assert packet.frame.shape == sample_frame.shape
        assert packet.frame_index == 0
        assert source.read() is None
        source.close()


class TestCameraSource:
    @patch("src.input.camera.cv2.VideoCapture")
    @patch("src.input.camera.get_settings")
    def test_open_read_close(self, mock_get_settings, mock_capture_cls, sample_frame):
        cfg = MagicMock(camera_index=0, camera_width=640, camera_height=480)
        mock_get_settings.return_value = cfg

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, sample_frame)
        mock_capture_cls.return_value = cap

        from src.input.camera import CameraSource

        source = CameraSource()
        source.open()
        packet = source.read()
        assert packet is not None
        assert packet.source_type == "camera"
        source.close()
        cap.release.assert_called_once()

    @patch("src.input.camera.cv2.VideoCapture")
    @patch("src.input.camera.get_settings")
    def test_frame_index_is_zero_based(self, mock_get_settings, mock_capture_cls, sample_frame):
        """First read() should return frame_index=0, not 1."""
        cfg = MagicMock(camera_index=0, camera_width=640, camera_height=480)
        mock_get_settings.return_value = cfg

        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.read.return_value = (True, sample_frame)
        mock_capture_cls.return_value = cap

        from src.input.camera import CameraSource

        source = CameraSource()
        source.open()
        p1 = source.read()
        p2 = source.read()
        assert p1.frame_index == 0
        assert p2.frame_index == 1


class TestVideoSource:
    @patch("src.input.video.cv2.VideoCapture")
    def test_open_read_properties_close(self, mock_capture_cls, sample_frame, tmp_path):
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {
            5: 24.0,  # CAP_PROP_FPS
            7: 120,   # CAP_PROP_FRAME_COUNT
        }.get(prop, 0)
        cap.read.side_effect = [(True, sample_frame), (False, None)]
        mock_capture_cls.return_value = cap

        from src.input.video import VideoSource

        video_path = tmp_path / "demo.mp4"
        video_path.write_bytes(b"placeholder")
        source = VideoSource(video_path)
        source.open()
        assert source.fps == 24.0
        assert source.total_frames == 120
        packet = source.read()
        assert packet is not None
        assert packet.source_type == "video"
        assert packet.metadata["fps"] == 24.0
        source.close()
        cap.release.assert_called_once()

    @patch("src.input.video.cv2.VideoCapture")
    def test_frame_index_is_zero_based(self, mock_capture_cls, sample_frame, tmp_path):
        """First read() should return frame_index=0, not 1."""
        cap = MagicMock()
        cap.isOpened.return_value = True
        cap.get.side_effect = lambda prop: {5: 30.0, 7: 100}.get(prop, 0)
        cap.read.side_effect = [(True, sample_frame), (True, sample_frame), (False, None)]
        mock_capture_cls.return_value = cap

        from src.input.video import VideoSource

        video_path = tmp_path / "test.mp4"
        video_path.write_bytes(b"placeholder")
        source = VideoSource(video_path)
        source.open()
        p1 = source.read()
        p2 = source.read()
        assert p1.frame_index == 0
        assert p2.frame_index == 1
