"""Pruebas de dominio sin GPU, descargas de modelos ni llamadas a servicios externos."""

import io
import logging
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from src.cloud_providers import (
    AssemblyAIPipeline,
    DeepgramPipeline,
    _response_json,
    extract_audio_for_cloud,
)
from src.config import (
    ProviderType,
    TimestampMode,
    TranscriptionConfig,
    normalize_provider_name,
)
from src.transcriber import WhisperXPipeline, create_transcription_pipeline
from src.utils import (
    AudioProcessingError,
    MissingTokenError,
    ProgressFileReader,
    TranscriptionError,
    create_zip_archive,
    format_duration_human,
    format_timestamp,
    make_progress_notifier,
    render_segments_text,
    resolve_deepgram_api_key,
    resolve_hf_token,
    save_transcription_file,
    validate_media_path,
    verify_admin_pin,
)


def cloud_config(provider=ProviderType.DEEPGRAM.value, **kwargs):
    return TranscriptionConfig.create_default(provider=provider, api_key="test-key", **kwargs)


def response(data=None, status=200):
    return Mock(status_code=status, json=Mock(return_value=data), text="test response")


class ConfigTests(unittest.TestCase):
    def test_provider_defaults(self):
        for provider, model in [
            (ProviderType.DEEPGRAM, "nova-2"),
            (ProviderType.ASSEMBLYAI, "best"),
        ]:
            with self.subTest(provider=provider):
                self.assertEqual(cloud_config(provider.value).model_name, model)

    def test_invalid_configuration_is_rejected(self):
        for changes in [
            {"provider": "unknown"},
            {"model_name": " "},
            {"batch_size": 0},
            {"batch_size": True},
            {"device": "invalid"},
            {"compute_type": "invalid"},
            {"timestamp_mode": "invalid"},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                TranscriptionConfig(**changes)

    def test_unknown_provider_default_is_rejected(self):
        with self.assertRaises(ValueError):
            TranscriptionConfig.create_default(provider="unknown")

    def test_provider_aliases_are_normalized(self):
        config_wx = TranscriptionConfig(provider="whisperx")
        self.assertEqual(config_wx.provider, ProviderType.WHISPERX.value)

        config_dg = TranscriptionConfig.create_default(provider="deepgram")
        self.assertEqual(config_dg.provider, ProviderType.DEEPGRAM.value)

        config_aai = TranscriptionConfig.create_default(provider="assemblyai")
        self.assertEqual(config_aai.provider, ProviderType.ASSEMBLYAI.value)

    def test_normalize_provider_name_function(self):
        self.assertEqual(normalize_provider_name("whisperx"), ProviderType.WHISPERX.value)
        self.assertEqual(normalize_provider_name("DEEPGRAM"), ProviderType.DEEPGRAM.value)
        self.assertEqual(normalize_provider_name("  assemblyai  "), ProviderType.ASSEMBLYAI.value)
        self.assertEqual(
            normalize_provider_name(ProviderType.WHISPERX.value), ProviderType.WHISPERX.value
        )
        for invalid in ["", "   ", "google", "openai", None, 123]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                normalize_provider_name(invalid)  # type: ignore[arg-type]

    def test_secrets_not_in_repr(self):
        config = TranscriptionConfig(api_key="api-secret", hf_token="hf-secret")
        self.assertNotIn("api-secret", repr(config))
        self.assertNotIn("hf-secret", repr(config))

    def test_string_timestamp_mode_is_normalized(self):
        self.assertIs(
            TranscriptionConfig(timestamp_mode="range").timestamp_mode, TimestampMode.RANGE
        )

    def test_explicit_hardware_is_preserved(self):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": torch}):
            config = TranscriptionConfig.create_default(
                provider=ProviderType.WHISPERX.value, device="cuda", compute_type="float16"
            )
        self.assertEqual((config.device, config.compute_type), ("cuda", "float16"))


class UtilityTests(unittest.TestCase):
    def test_timestamps(self):
        for value, expected in [
            (None, "00:00:00"),
            (-1, "00:00:00"),
            (float("nan"), "00:00:00"),
            (float("inf"), "00:00:00"),
            (3661.9, "01:01:01"),
        ]:
            with self.subTest(value=value):
                self.assertEqual(format_timestamp(value), expected)

    def test_format_duration_human(self):
        for value, expected in [
            (None, "0.0s"),
            (-5, "0.0s"),
            (float("nan"), "0.0s"),
            (12.34, "12.3 segundos"),
            (59.9, "59.9 segundos"),
            (60.0, "1 min 0 s (60.0s)"),
            (713.5, "11 min 53 s (713.5s)"),
        ]:
            with self.subTest(value=value):
                self.assertEqual(format_duration_human(value), expected)

    def test_render_modes_and_speaker(self):
        segments = [{"text": " Hola ", "start": 1, "end": 3, "speaker": "A"}]
        self.assertEqual(render_segments_text(segments), "[00:00:01] [A] Hola\n")
        self.assertEqual(
            render_segments_text(segments, TimestampMode.RANGE, False),
            "[00:00:01 - 00:00:03] Hola\n",
        )
        self.assertEqual(render_segments_text(segments, TimestampMode.NONE), "[A] Hola\n")
        self.assertEqual(render_segments_text([{"text": " "}]), "")

    def test_media_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for path in [root, root / "missing"]:
                with self.assertRaises(AudioProcessingError):
                    validate_media_path(path)
            media = root / "audio.wav"
            media.touch()
            with self.assertRaises(AudioProcessingError):
                validate_media_path(media)
            media.write_bytes(b"audio")
            self.assertEqual(validate_media_path(media), media)

    def test_atomic_write_preserves_previous_output_on_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "audio.wav"
            output = save_transcription_file("anterior", media)
            with (
                patch.object(Path, "replace", side_effect=PermissionError),
                self.assertRaises(TranscriptionError),
            ):
                save_transcription_file("nuevo", media)
            self.assertEqual(output.read_text(encoding="utf-8"), "anterior")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_utf8_output_and_nested_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = save_transcription_file("áéñ", Path("audio.wav"), Path(directory) / "nested")
            self.assertEqual(output.read_text(encoding="utf-8"), "áéñ")

    def test_secret_priority(self):
        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": " environment "}):
            self.assertEqual(resolve_deepgram_api_key(" explicit "), "explicit")
            self.assertEqual(resolve_deepgram_api_key(" "), "environment")

    def test_env_file_parsing_with_whitespace_and_comments(self):
        env_content = (
            "# Archivo de configuración\n"
            "  HF_TOKEN = \"mi-token-con-comillas\" # comentario inline\n"
            "DEEPGRAM_API_KEY = mi-token-sin-comillas # comentario inline\n"
            "ASSEMBLYAI_API_KEY = 'mi-token-single' \n"
            "IGNORAR_SIN_VALOR = \n"
        )
        with (
            patch.dict("os.environ", {}, clear=True),
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", unittest.mock.mock_open(read_data=env_content)),
        ):
            self.assertEqual(resolve_hf_token(), "mi-token-con-comillas")
            self.assertEqual(resolve_deepgram_api_key(), "mi-token-sin-comillas")

    def test_admin_pin(self):
        with patch("src.utils.resolve_admin_pin", return_value="secreto-á"):
            self.assertTrue(verify_admin_pin("secreto-á"))
            self.assertFalse(verify_admin_pin("incorrecto"))
            self.assertFalse(verify_admin_pin(None))
        with patch("src.utils.resolve_admin_pin", return_value=None):
            self.assertFalse(verify_admin_pin("anything"))

    def test_progress_is_clamped_and_callback_failure_isolated(self):
        callback = Mock(side_effect=RuntimeError)
        notify = make_progress_notifier(logging.getLogger("test"), callback)
        for value in [0.5, 0.2, 2]:
            notify(value, "stage")
        self.assertEqual([call.args[0] for call in callback.call_args_list], [0.5, 0.5, 1])

    def test_upload_reader_reports_bytes(self):
        notify = Mock()
        reader = ProgressFileReader(io.BytesIO(b"abcde"), 5, notify, 0.1, 0.5, "upload")
        self.assertEqual(reader.read(2) + reader.read(), b"abcde")
        self.assertEqual(reader.bytes_read, 5)
        self.assertEqual(len(reader), 5)
        self.assertEqual(notify.call_args.args[0], 0.5)

    def test_create_zip_archive_success(self):
        with tempfile.TemporaryDirectory() as directory:
            dir_path = Path(directory)
            f1 = dir_path / "transcripcion_1.txt"
            f2 = dir_path / "transcripcion_2.txt"
            f1.write_text("texto 1", encoding="utf-8")
            f2.write_text("texto 2", encoding="utf-8")

            zip_output = dir_path / "salida" / "lote.zip"
            result_zip = create_zip_archive([f1, f2], zip_output)
            self.assertEqual(result_zip, zip_output)
            self.assertTrue(zip_output.is_file())

            with zipfile.ZipFile(zip_output, "r") as zf:
                self.assertEqual(sorted(zf.namelist()), ["transcripcion_1.txt", "transcripcion_2.txt"])
                self.assertEqual(zf.read("transcripcion_1.txt").decode("utf-8"), "texto 1")
                self.assertEqual(zf.read("transcripcion_2.txt").decode("utf-8"), "texto 2")

    def test_create_zip_archive_atomic_cleanup_on_error(self):
        with tempfile.TemporaryDirectory() as directory:
            dir_path = Path(directory)
            f1 = dir_path / "test.txt"
            f1.write_text("contenido", encoding="utf-8")
            zip_output = dir_path / "fallo.zip"

            with (
                patch.object(Path, "replace", side_effect=PermissionError("acceso denegado")),
                self.assertRaises(PermissionError),
            ):
                create_zip_archive([f1], zip_output)

            self.assertFalse(zip_output.exists())
            self.assertFalse(zip_output.with_suffix(".tmp.zip").exists())


class CloudTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.media = Path(self.directory.name) / "audio.wav"
        self.media.write_bytes(b"audio")
        self.deepgram = DeepgramPipeline(cloud_config())
        self.assembly = AssemblyAIPipeline(cloud_config(ProviderType.ASSEMBLYAI.value))

    def test_factory(self):
        self.assertIsInstance(create_transcription_pipeline(cloud_config()), DeepgramPipeline)
        self.assertIsInstance(
            create_transcription_pipeline(cloud_config(ProviderType.ASSEMBLYAI.value)),
            AssemblyAIPipeline,
        )

    def test_missing_cloud_credentials(self):
        for pipeline, resolver in [
            (DeepgramPipeline, "resolve_deepgram_api_key"),
            (AssemblyAIPipeline, "resolve_assemblyai_api_key"),
        ]:
            with (
                self.subTest(pipeline=pipeline),
                patch(f"src.cloud_providers.{resolver}", return_value=None),
                self.assertRaises(TranscriptionError),
            ):
                pipeline(TranscriptionConfig())

    def test_response_json_validation(self):
        for invalid in [None, [], "text"]:
            with self.subTest(invalid=invalid), self.assertRaises(TranscriptionError):
                _response_json(response(invalid))
        with self.assertRaises(TranscriptionError):
            _response_json(Mock(json=Mock(side_effect=ValueError)))

    def test_deepgram_language_parameters(self):
        self.assertEqual(self.deepgram._build_query_params()["detect_language"], "true")
        params = DeepgramPipeline(cloud_config(language="es"))._build_query_params()
        self.assertEqual(params["language"], "es")
        self.assertNotIn("detect_language", params)

    def test_deepgram_diarization_and_duration(self):
        pipeline = DeepgramPipeline(cloud_config(enable_diarization=True))
        segments = pipeline._extract_segments(
            {
                "words": [
                    {"word": "uno", "start": 0, "end": 1, "speaker": 0},
                    {"punctuated_word": "Dos.", "start": 2, "end": 3, "speaker": 1},
                    {"word": "tres", "start": 28, "end": 29, "speaker": 1},
                ]
            }
        )
        self.assertEqual(len(segments), 3)
        self.assertEqual(segments[0]["speaker"], "SPEAKER_00")
        self.assertEqual(segments[1]["text"], "Dos.")

    def test_deepgram_process(self):
        data = {
            "results": {
                "channels": [{"detected_language": "es", "alternatives": [{"transcript": "Hola"}]}]
            }
        }
        with patch("src.cloud_providers.requests.post", return_value=response(data)):
            result = self.deepgram.process(self.media)
        self.assertEqual(result.detected_language, "es")
        self.assertIn("Hola", result.output_path.read_text(encoding="utf-8"))

    def test_deepgram_http_and_connection_errors(self):
        for status in [401, 403, 429, 500]:
            with (
                self.subTest(status=status),
                patch("src.cloud_providers.requests.post", return_value=response(status=status)),
                self.assertRaises(TranscriptionError),
            ):
                self.deepgram.process(self.media)
        with (
            patch("src.cloud_providers.requests.post", side_effect=requests.Timeout),
            self.assertRaises(TranscriptionError),
        ):
            self.deepgram.process(self.media)

    def test_deepgram_missing_response_fields(self):
        with (
            patch("src.cloud_providers.requests.post", return_value=response({})),
            self.assertRaises(TranscriptionError),
        ):
            self.deepgram.process(self.media)

    def test_assembly_segment_start_after_pause(self):
        segments = self.assembly._extract_segments(
            {
                "words": [
                    {"text": "Hola.", "start": 1000, "end": 2000},
                    {"text": "Adiós.", "start": 15000, "end": 16000},
                ]
            }
        )
        self.assertEqual([segment["start"] for segment in segments], [1, 15])

    def test_assembly_diarization(self):
        pipeline = AssemblyAIPipeline(
            cloud_config(ProviderType.ASSEMBLYAI.value, enable_diarization=True)
        )
        segments = pipeline._extract_segments(
            {"utterances": [{"text": "Hola", "start": 1000, "end": 2000, "speaker": "B"}]}
        )
        self.assertEqual(
            segments[0], {"text": "Hola", "start": 1, "end": 2, "speaker": "SPEAKER_B"}
        )

    def test_assembly_process(self):
        with (
            patch(
                "src.cloud_providers.requests.post",
                side_effect=[
                    response({"upload_url": "https://example.com/audio"}),
                    response({"id": "job"}),
                ],
            ),
            patch(
                "src.cloud_providers.requests.get",
                return_value=response(
                    {"status": "completed", "text": "Hola", "language_code": "es"}
                ),
            ),
            patch("src.cloud_providers.time.sleep"),
        ):
            result = self.assembly.process(self.media)
        self.assertIn("Hola", result.formatted_text)
        self.assertTrue(result.output_path.is_file())

    def test_assembly_terminal_errors_fail_immediately(self):
        for status in [401, 403, 404]:
            with (
                self.subTest(status=status),
                patch(
                    "src.cloud_providers.requests.get", return_value=response(status=status)
                ) as get,
                patch("src.cloud_providers.time.sleep"),
                self.assertRaises(TranscriptionError),
            ):
                self.assembly._poll_completion("job", {}, Mock())
            get.assert_called_once()

    def test_assembly_transient_error_is_retried(self):
        with (
            patch(
                "src.cloud_providers.requests.get",
                side_effect=[
                    requests.Timeout(),
                    response(status=503),
                    response({"status": "completed"}),
                ],
            ) as get,
            patch("src.cloud_providers.time.sleep"),
        ):
            self.assertEqual(
                self.assembly._poll_completion("job", {}, Mock())["status"], "completed"
            )
        self.assertEqual(get.call_count, 3)

    def test_assembly_processing_error_and_unknown_status(self):
        for data in [{"status": "error", "error": "failure"}, {"status": "unexpected"}]:
            with (
                self.subTest(data=data),
                patch("src.cloud_providers.requests.get", return_value=response(data)),
                patch("src.cloud_providers.time.sleep"),
                self.assertRaises(TranscriptionError),
            ):
                self.assembly._poll_completion("job", {}, Mock())

    def test_assembly_timeout_uses_wall_clock(self):
        with (
            patch("src.cloud_providers.time.monotonic", side_effect=[0, 3601]),
            patch("src.cloud_providers.requests.get") as get,
            self.assertRaises(TranscriptionError),
        ):
            self.assembly._poll_completion("job", {}, Mock())
        get.assert_not_called()

    def test_invalid_file_never_contacts_provider(self):
        for pipeline in [self.deepgram, self.assembly]:
            with (
                self.subTest(pipeline=pipeline),
                patch("src.cloud_providers.requests.post") as post,
                self.assertRaises(AudioProcessingError),
            ):
                pipeline.process(self.directory.name)
            post.assert_not_called()

    def test_extract_audio_for_cloud_skips_audio_files(self):
        audio_file = Path("test.mp3")
        path, is_temp = extract_audio_for_cloud(audio_file)
        self.assertEqual(path, audio_file)
        self.assertFalse(is_temp)

    def test_extract_audio_for_cloud_video_success(self):
        video_file = Path("test.mp4")
        mock_proc = Mock(returncode=0)
        with (
            patch("subprocess.run", return_value=mock_proc),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_size=1024)),
        ):
            path, is_temp = extract_audio_for_cloud(video_file)
            self.assertTrue(is_temp)
            self.assertTrue(str(path).endswith(".m4a"))
            path.unlink(missing_ok=True)

    def test_extract_audio_for_cloud_video_fallback_on_failure(self):
        video_file = Path("test.mp4")
        mock_proc = Mock(returncode=1)
        with patch("subprocess.run", return_value=mock_proc):
            path, is_temp = extract_audio_for_cloud(video_file)
            self.assertEqual(path, video_file)
            self.assertFalse(is_temp)


class WhisperTests(unittest.TestCase):
    def setUp(self):
        self.config = TranscriptionConfig(
            provider=ProviderType.WHISPERX.value, model_name="base", device="cpu"
        )

    def test_diarization_requires_token(self):
        with (
            patch("src.transcriber.resolve_hf_token", return_value=None),
            self.assertRaises(MissingTokenError),
        ):
            WhisperXPipeline(
                TranscriptionConfig(
                    provider=ProviderType.WHISPERX.value,
                    model_name="base",
                    device="cpu",
                    enable_diarization=True,
                )
            )

    def test_whisper_success_empty_audio_and_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "audio.wav"
            media.write_bytes(b"audio")
            for empty in [False, True]:
                segments = [] if empty else [{"text": "Hola", "start": 1, "end": 2}]
                whisper = Mock()
                whisper.load_model.return_value.transcribe.return_value = {
                    "language": "es",
                    "segments": segments,
                }
                whisper.load_align_model.return_value = (Mock(), {})
                whisper.align.return_value = {"segments": segments}
                with (
                    self.subTest(empty=empty),
                    patch.dict("sys.modules", {"whisperx": whisper}),
                    patch("src.transcriber.clear_gpu_vram") as cleanup,
                ):
                    result = WhisperXPipeline(self.config).process(media)
                self.assertEqual(result.raw_segments, segments)
                self.assertTrue(cleanup.called)
                if empty:
                    whisper.load_align_model.assert_not_called()
            whisper.load_audio.side_effect = RuntimeError("decode failure")
            with (
                patch.dict("sys.modules", {"whisperx": whisper}),
                patch("src.transcriber.clear_gpu_vram") as cleanup,
                self.assertRaises(TranscriptionError),
            ):
                WhisperXPipeline(self.config).process(media)
            cleanup.assert_called_once()

    def test_whisper_diarization_execution_and_gated_error(self):
        config = TranscriptionConfig(
            provider=ProviderType.WHISPERX.value,
            model_name="base",
            device="cpu",
            enable_diarization=True,
            hf_token="test_hf_token",
        )
        pipeline = WhisperXPipeline(config)
        aligned_result = {"segments": [{"text": "Hola", "start": 0, "end": 1}]}

        diarize_pipeline_mock = Mock(return_value=[])
        diarize_module = SimpleNamespace(
            DiarizationPipeline=Mock(return_value=diarize_pipeline_mock),
            assign_word_speakers=Mock(
                return_value={"segments": [{"text": "Hola", "speaker": "SPEAKER_00"}]}
            ),
        )
        with patch.dict("sys.modules", {"whisperx.diarize": diarize_module}):
            segments = pipeline._run_diarization(audio=None, aligned_result=aligned_result, device="cpu")
            self.assertEqual(segments[0]["speaker"], "SPEAKER_00")

        diarize_module_error = SimpleNamespace(
            DiarizationPipeline=Mock(
                side_effect=RuntimeError("401 Client Error: Repository is gated")
            ),
            assign_word_speakers=Mock(),
        )
        with (
            patch.dict("sys.modules", {"whisperx.diarize": diarize_module_error}),
            self.assertRaises(TranscriptionError) as err,
        ):
            pipeline._run_diarization(audio=None, aligned_result=aligned_result, device="cpu")
        self.assertIn("pyannote/speaker-diarization-3.1", str(err.exception))


if __name__ == "__main__":
    unittest.main()
