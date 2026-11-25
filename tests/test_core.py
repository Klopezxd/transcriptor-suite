"""Pruebas de dominio sin GPU, descargas de modelos ni llamadas a servicios externos."""

import io
import logging
import os
import subprocess
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
    clear_gpu_vram,
    create_zip_archive,
    format_duration_human,
    format_timestamp,
    hash_admin_pin,
    make_progress_notifier,
    render_segments_text,
    resolve_deepgram_api_key,
    resolve_hf_token,
    save_consolidated_batch_file,
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
            (ProviderType.DEEPGRAM, "nova-3"),
            (ProviderType.ASSEMBLYAI, "assemblyai-stt-3-7"),
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

    def test_save_consolidated_batch_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out_file = save_consolidated_batch_file(
                "texto consolidado", directory, timestamp_str="20261016_120000"
            )
            self.assertTrue(out_file.is_file())
            self.assertEqual(out_file.name, "transcripcion_consolidada_lote_20261016_120000.txt")
            self.assertEqual(out_file.read_text(encoding="utf-8"), "texto consolidado")

    def test_save_consolidated_batch_file_atomic_failure_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(Path, "replace", side_effect=PermissionError),
                self.assertRaises(TranscriptionError),
            ):
                save_consolidated_batch_file("texto", directory)
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

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

    def test_assembly_submit_job_uses_speech_models_list(self):
        pipeline = AssemblyAIPipeline(
            cloud_config(ProviderType.ASSEMBLYAI.value, model_name="assemblyai-stt-3-7")
        )
        with patch("src.cloud_providers.requests.post", return_value=response({"id": "job-123"})) as post_mock:
            job_id = pipeline._submit_transcript_job("https://example.com/audio.mp3", {"authorization": "test"})
            self.assertEqual(job_id, "job-123")
            payload = post_mock.call_args[1]["json"]
            self.assertIn("speech_models", payload)
            self.assertNotIn("speech_model", payload)
            self.assertEqual(payload["speech_models"], ["assemblyai-stt-3-7", "universal-2"])

        # Verificar retrocompatibilidad con universal-3-5-pro
        legacy_pipeline = AssemblyAIPipeline(
            cloud_config(ProviderType.ASSEMBLYAI.value, model_name="universal-3-5-pro")
        )
        with patch("src.cloud_providers.requests.post", return_value=response({"id": "job-456"})) as post_mock_legacy:
            legacy_pipeline._submit_transcript_job("https://example.com/audio.mp3", {"authorization": "test"})
            legacy_payload = post_mock_legacy.call_args[1]["json"]
            self.assertEqual(legacy_payload["speech_models"], ["universal-3-5-pro", "universal-2"])

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

    def test_extract_audio_for_cloud_video_flac_fallback(self):
        video_file = Path("test.mkv")
        mock_fail = Mock(returncode=1)
        mock_succ = Mock(returncode=0)
        with (
            patch("subprocess.run", side_effect=[mock_fail, mock_succ]),
            patch("pathlib.Path.is_file", return_value=True),
            patch("pathlib.Path.stat", return_value=SimpleNamespace(st_size=2048)),
        ):
            path, is_temp = extract_audio_for_cloud(video_file)
            self.assertTrue(is_temp)
            self.assertTrue(str(path).endswith(".flac"))
            path.unlink(missing_ok=True)


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

    def test_cpu_float16_is_normalized_to_int8_and_device_auto(self):
        cfg = TranscriptionConfig(
            provider=ProviderType.WHISPERX.value,
            model_name="base",
            device="cpu",
            compute_type="float16",
        )
        self.assertEqual(cfg.compute_type, "int8")
        self.assertFalse(cfg.device_auto)

        torch_mod = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))
        with patch.dict("sys.modules", {"torch": torch_mod}):
            auto_cfg = TranscriptionConfig.create_default(provider=ProviderType.WHISPERX.value)
        self.assertTrue(auto_cfg.device_auto)
        self.assertFalse(cloud_config().device_auto)

    def test_phase3_alignment_unknown_language_and_failure_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "audio.wav"
            media.write_bytes(b"audio")
            segments = [{"text": "Hola mundo", "start": 0.5, "end": 1.5}]

            whisper_unknown = Mock()
            whisper_unknown.load_model.return_value.transcribe.return_value = {
                "language": "unknown",
                "segments": segments,
            }
            with (
                patch.dict("sys.modules", {"whisperx": whisper_unknown}),
                patch("src.transcriber.clear_gpu_vram"),
            ):
                res_unknown = WhisperXPipeline(self.config).process(media)
            self.assertEqual(res_unknown.raw_segments, segments)
            whisper_unknown.load_align_model.assert_not_called()

            whisper_fail = Mock()
            whisper_fail.load_model.return_value.transcribe.return_value = {
                "language": "eu",
                "segments": segments,
            }
            whisper_fail.load_align_model.side_effect = ValueError("No default align-model")
            with (
                patch.dict("sys.modules", {"whisperx": whisper_fail}),
                patch("src.transcriber.clear_gpu_vram"),
            ):
                res_fail = WhisperXPipeline(self.config).process(media)
            self.assertEqual(res_fail.raw_segments, segments)
            self.assertIn("Hola mundo", res_fail.formatted_text)

    def test_whisper_cuda_oom_reduces_batch_and_falls_back_to_cpu(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "audio.wav"
            media.write_bytes(b"audio")

            calls: list[tuple[str, str, int]] = []
            segments = [{"text": "Recuperado", "start": 0.0, "end": 1.0}]

            def fake_load_model(model_name, device, compute_type, language=None):
                def fake_transcribe(audio, batch_size):
                    calls.append((device, compute_type, batch_size))
                    if device == "cuda":
                        raise RuntimeError("CUDA out of memory")
                    return {"language": "es", "segments": segments}

                return SimpleNamespace(transcribe=fake_transcribe)

            whisper = Mock()
            whisper.load_model.side_effect = fake_load_model
            whisper.load_align_model.return_value = (Mock(), {})
            whisper.align.return_value = {"segments": segments}

            torch_mod = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True))
            with patch.dict("sys.modules", {"torch": torch_mod, "whisperx": whisper}):
                cfg = TranscriptionConfig.create_default(
                    provider=ProviderType.WHISPERX.value, model_name="base"
                )
                with patch("src.transcriber.clear_gpu_vram"):
                    res = WhisperXPipeline(cfg).process(media)

            self.assertEqual(res.raw_segments, segments)
            self.assertEqual(
                calls,
                [
                    ("cuda", "int8", 16),
                    ("cuda", "int8", 4),
                    ("cuda", "int8", 1),
                    ("cpu", "int8", 4),
                ],
            )

    def test_whisper_diarization_cuda_oom_falls_back_to_cpu(self):
        cfg = TranscriptionConfig(
            provider=ProviderType.WHISPERX.value,
            model_name="base",
            device="cuda",
            enable_diarization=True,
            hf_token="test_hf_token",
        )
        pipeline = WhisperXPipeline(cfg)
        aligned = {"segments": [{"text": "Hola", "start": 0, "end": 1}]}
        seen_devices: list[str] = []

        def fake_diarize_pipeline(**kwargs):
            seen_devices.append(kwargs["device"])
            if kwargs["device"] == "cuda":
                raise RuntimeError("CUDA out of memory")
            return Mock(return_value=[])

        diarize_mod = SimpleNamespace(
            DiarizationPipeline=fake_diarize_pipeline,
            assign_word_speakers=Mock(
                return_value={"segments": [{"text": "Hola", "speaker": "SPEAKER_00"}]}
            ),
        )
        with (
            patch.dict("sys.modules", {"whisperx.diarize": diarize_mod}),
            patch("src.transcriber.clear_gpu_vram"),
        ):
            segs = pipeline._run_diarization(audio=None, aligned_result=aligned, device="cuda")

        self.assertEqual(seen_devices, ["cuda", "cpu"])
        self.assertEqual(segs[0]["speaker"], "SPEAKER_00")

    def test_utility_and_cloud_hardening_regressions(self):
        # 1. validate_media_path con allowed_roots, symlinks y sin filtrar rutas absolutas
        with tempfile.TemporaryDirectory() as allowed_dir, tempfile.TemporaryDirectory() as outside_dir:
            allowed_root = Path(allowed_dir)
            outside_file = Path(outside_dir) / "secreto.wav"
            outside_file.write_bytes(b"audio")
            bad_ext = allowed_root / "notes.txt"
            bad_ext.write_text("no media", encoding="utf-8")

            with self.assertRaises(AudioProcessingError) as ctx_out:
                validate_media_path(outside_file, allowed_roots=[allowed_root])
            self.assertNotIn(str(Path(outside_dir).resolve()), str(ctx_out.exception))

            with self.assertRaises(AudioProcessingError):
                validate_media_path(bad_ext, allowed_roots=[allowed_root])

            link_path = allowed_root / "escape.wav"
            try:
                os.symlink(outside_file, link_path)
            except OSError:
                link_path = None
            if link_path is not None:
                with self.assertRaises(AudioProcessingError):
                    validate_media_path(link_path, allowed_roots=[allowed_root])

        # 2. ProgressFileReader seek/tell/readable y make_progress_notifier NaN
        notify = Mock()
        reader = ProgressFileReader(io.BytesIO(b"abcdef"), 6, notify, 0.0, 1.0, "upload")
        self.assertTrue(reader.readable())
        self.assertEqual(reader.read(3), b"abc")
        self.assertEqual(reader.tell(), 3)
        self.assertEqual(reader.seek(0), 0)
        self.assertEqual(reader.bytes_read, 0)
        self.assertEqual(reader.read(), b"abcdef")

        safe_cb = Mock()
        safe_notify = make_progress_notifier(logging.getLogger("test_nan"), safe_cb)
        safe_notify(0.4, "ok")
        safe_notify(float("nan"), "nan")
        self.assertEqual(safe_cb.call_args_list[-1].args[0], 0.4)

        # 3. render_segments_text con timestamps None y create_zip_archive deduplicado
        self.assertEqual(
            render_segments_text([{"text": "Hola", "start": None, "end": None}]),
            "[00:00:00] Hola\n",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "d1").mkdir()
            (root / "d2").mkdir()
            f1 = root / "d1" / "clase_TRANSCRIPCION.txt"
            f2 = root / "d2" / "clase_TRANSCRIPCION.txt"
            f1.write_text("uno", encoding="utf-8")
            f2.write_text("dos", encoding="utf-8")
            zip_out = root / "batch.zip"
            create_zip_archive([f1, f2], zip_out)
            with zipfile.ZipFile(zip_out, "r") as zf:
                self.assertEqual(
                    sorted(zf.namelist()),
                    ["clase_TRANSCRIPCION (2).txt", "clase_TRANSCRIPCION.txt"],
                )

        # 4. ADMIN_PIN_HASH scrypt y _resolve_secret con export
        hashed = hash_admin_pin("mi_pin_seguro")
        with patch.dict("os.environ", {"ADMIN_PIN_HASH": hashed, "ADMIN_PIN": "otro"}):
            self.assertTrue(verify_admin_pin("mi_pin_seguro"))
            self.assertFalse(verify_admin_pin("otro"))
        with patch.dict("os.environ", {"ADMIN_PIN_HASH": "scrypt$bad$hex", "ADMIN_PIN": "1234"}):
            self.assertFalse(verify_admin_pin("1234"))

        with tempfile.TemporaryDirectory() as env_dir, tempfile.TemporaryDirectory() as other_cwd:
            env_file = Path(env_dir) / ".env"
            env_file.write_text('export DEEPGRAM_API_KEY="dg_from_root"\n', encoding="utf-8")
            prev_cwd = os.getcwd()
            try:
                os.chdir(other_cwd)
                with (
                    patch.dict("os.environ", {}, clear=True),
                    patch("src.utils.PROJECT_ROOT", Path(env_dir)),
                ):
                    self.assertEqual(resolve_deepgram_api_key(), "dg_from_root")
            finally:
                os.chdir(prev_cwd)

        # 5. clear_gpu_vram ejecuta empty_cache e ipc_collect aunque synchronize falle
        cuda_mock = SimpleNamespace(
            is_available=lambda: True,
            synchronize=Mock(side_effect=RuntimeError("post-oom")),
            empty_cache=Mock(),
            ipc_collect=Mock(),
        )
        with patch.dict("sys.modules", {"torch": SimpleNamespace(cuda=cuda_mock)}):
            clear_gpu_vram()
        cuda_mock.empty_cache.assert_called_once()
        cuda_mock.ipc_collect.assert_called_once()

    def test_cloud_speaker_parsing_retries_and_temp_cleanup(self):
        # Deepgram speaker str/float/None y start=None
        dg = DeepgramPipeline(cloud_config(enable_diarization=True))
        segs = dg._extract_segments(
            {
                "words": [
                    {"word": "hola", "start": None, "end": 1, "speaker": "0"},
                    {"word": "mundo", "start": 2, "end": 3, "speaker": "A"},
                    {"word": "fin", "start": 4, "end": 5, "speaker": 2.0},
                ]
            }
        )
        self.assertEqual([s["speaker"] for s in segs], ["SPEAKER_00", "SPEAKER_A", "SPEAKER_02"])

        # Deepgram 429 reintenta con Retry-After y rebobina archivo
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "audio.wav"
            media.write_bytes(b"payload")
            seen_bodies: list[bytes] = []

            def fake_post(url, **kwargs):
                seen_bodies.append(kwargs["data"].read())
                if len(seen_bodies) == 1:
                    r = response(status=429)
                    r.headers = {"Retry-After": "0"}
                    return r
                return response(
                    {"results": {"channels": [{"detected_language": "es", "alternatives": [{"transcript": "Ok"}]}]}}
                )

            with (
                patch("src.cloud_providers.requests.post", side_effect=fake_post),
                patch("src.cloud_providers.time.sleep") as sleep_mock,
            ):
                res = dg.process(media)
            self.assertEqual(seen_bodies, [b"payload", b"payload"])
            sleep_mock.assert_called_once()
            self.assertIn("Ok", res.formatted_text)

        # AssemblyAI elimina temporal inmediatamente tras _upload_media (antes de polling)
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "video.webm"
            video.write_bytes(b"webm")
            temp_audio = Path(directory) / "temp_extracted.m4a"
            temp_audio.write_bytes(b"m4a")
            aai = AssemblyAIPipeline(cloud_config(ProviderType.ASSEMBLYAI.value))

            def fake_get(url, **kwargs):
                self.assertFalse(temp_audio.exists())
                return response({"status": "completed", "text": "Listo", "language_code": "es"})

            with (
                patch("src.cloud_providers.extract_audio_for_cloud", return_value=(temp_audio, True)),
                patch(
                    "src.cloud_providers.requests.post",
                    side_effect=[
                        response({"upload_url": "https://example.com/a"}),
                        response({"id": "job-1"}),
                    ],
                ),
                patch("src.cloud_providers.requests.get", side_effect=fake_get),
                patch("src.cloud_providers.time.sleep") as poll_sleep,
            ):
                res_aai = aai.process(video)
            self.assertIn("Listo", res_aai.formatted_text)
            poll_sleep.assert_not_called()

        # AssemblyAI polling respeta Retry-After ante 429
        r429 = response(status=429)
        r429.headers = {"Retry-After": "3"}
        with (
            patch(
                "src.cloud_providers.requests.get",
                side_effect=[r429, response({"status": "completed"})],
            ),
            patch("src.cloud_providers.time.sleep") as retry_sleep,
        ):
            aai._poll_completion("job-429", {}, Mock())
        retry_sleep.assert_called_once()
        self.assertGreaterEqual(retry_sleep.call_args.args[0], 3.0)

        # extract_audio_for_cloud con .webm, FileNotFoundError y TimeoutExpired
        webm = Path("clip.webm")
        with patch("subprocess.run", side_effect=FileNotFoundError("no ffmpeg")) as run_mock:
            path_nf, is_tmp_nf = extract_audio_for_cloud(webm)
        self.assertEqual((path_nf, is_tmp_nf), (webm, False))
        run_mock.assert_called_once()

        with patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=180),
        ) as run_to:
            path_to, is_tmp_to = extract_audio_for_cloud(webm)
        self.assertEqual((path_to, is_tmp_to), (webm, False))
        run_to.assert_called_once()

        # AssemblyAI upload no invoca smooth_progress_stage concurrente y Deepgram usa gate
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "audio.wav"
            media.write_bytes(b"123456")
            with (
                patch("src.cloud_providers.smooth_progress_stage") as smooth_mock,
                patch(
                    "src.cloud_providers.requests.post",
                    return_value=response({"upload_url": "https://example.com/u"}),
                ),
            ):
                aai._upload_media(media, {}, Mock())
            smooth_mock.assert_not_called()

            gate_holder: list[object] = []

            def capture_smooth(*args, **kwargs):
                gate_holder.append(kwargs.get("gate"))
                from contextlib import nullcontext

                return nullcontext()

            def fake_dg_post(url, **kwargs):
                self.assertFalse(gate_holder[0]())
                kwargs["data"].read()
                self.assertTrue(gate_holder[0]())
                return response(
                    {"results": {"channels": [{"detected_language": "es", "alternatives": [{"transcript": "Ok"}]}]}}
                )

            with (
                patch("src.cloud_providers.smooth_progress_stage", side_effect=capture_smooth),
                patch("src.cloud_providers.requests.post", side_effect=fake_dg_post),
            ):
                dg.process(media)
            self.assertEqual(len(gate_holder), 1)
            self.assertTrue(callable(gate_holder[0]))


if __name__ == "__main__":
    unittest.main()
