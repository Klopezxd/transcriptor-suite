"""Regresiones de eventos web y aislamiento de credenciales por sesión."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import app
from src.auth import LoginAttemptLimiter
from src.config import ProviderType


class AppTests(unittest.TestCase):
    def setUp(self):
        limiter = patch("app.admin_login_limiter", LoginAttemptLimiter())
        limiter.start()
        self.addCleanup(limiter.stop)

    def test_initial_load_always_requires_login(self):
        with patch("app.verify_admin_pin") as verify:
            result = app.initialize_guest_session(ProviderType.DEEPGRAM.value, False)
        self.assertFalse(result[0])
        self.assertEqual(len(result), 8)
        verify.assert_not_called()

    def test_repeated_failures_show_retry_message(self):
        request = SimpleNamespace(client=SimpleNamespace(host="test-client"))
        with patch("app.verify_admin_pin", return_value=False) as verify:
            for _ in range(6):
                result = app.process_admin_login(
                    "wrong", ProviderType.DEEPGRAM.value, False, request
                )
        self.assertFalse(result[0])
        self.assertIn("Demasiados intentos", result[-2]["value"])
        self.assertEqual(result[-1], "")
        self.assertEqual(verify.call_count, 5)

    def test_guest_cannot_use_server_cloud_credentials(self):
        for provider in [ProviderType.DEEPGRAM, ProviderType.ASSEMBLYAI]:
            with (
                self.subTest(provider=provider),
                patch.dict(
                    "os.environ", {"DEEPGRAM_API_KEY": "server", "ASSEMBLYAI_API_KEY": "server"}
                ),
            ):
                _, api_key, error = app._resolve_provider_credentials(
                    provider.value, False, "", "", "", False
                )
                self.assertIsNone(api_key)
                self.assertIsNotNone(error)

    def test_admin_can_resolve_server_credentials(self):
        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "server"}):
            _, api_key, error = app._resolve_provider_credentials(
                ProviderType.DEEPGRAM.value, False, "", "", "", True
            )
        self.assertEqual(api_key, "server")
        self.assertIsNone(error)

    def test_guest_personal_key_takes_priority(self):
        with patch.dict("os.environ", {"DEEPGRAM_API_KEY": "server"}):
            _, api_key, error = app._resolve_provider_credentials(
                ProviderType.DEEPGRAM.value, False, "", "personal", "", False
            )
        self.assertEqual(api_key, "personal")
        self.assertIsNone(error)

    def test_missing_media_does_not_start_pipeline(self):
        with patch("app.create_transcription_pipeline") as factory:
            result = app.process_transcription(None, "", "", False, "")
        factory.assert_not_called()
        self.assertEqual(len(result), 5)
        self.assertFalse(result[1].interactive)
        self.assertFalse(result[2].visible)
        self.assertTrue(result[4].interactive)

    def test_unknown_provider_returns_controlled_error(self):
        result = app.process_transcription("audio.wav", "unknown", "model", False, "")
        self.assertEqual(len(result), 5)
        self.assertIn("Error de transcripción", result[0])
        self.assertFalse(result[2].visible)
        self.assertTrue(result[4].interactive)

    def test_provider_response_is_escaped_in_html(self):
        result = SimpleNamespace(
            formatted_text="Hola",
            detected_language='<img src=x onerror="alert(1)">',
            output_path=Path("audio_TRANSCRIPCION.txt"),
        )
        pipeline = Mock(process=Mock(return_value=result))
        with patch("app.create_transcription_pipeline", return_value=pipeline):
            output = app.process_transcription(
                "audio.wav",
                ProviderType.DEEPGRAM.value,
                "nova-2",
                False,
                "",
                deepgram_key_input="personal",
                progress=Mock(),
            )
        self.assertEqual(len(output), 5)
        self.assertIn("&lt;IMG", output[3])
        self.assertNotIn("<IMG", output[3])
        self.assertTrue(output[1].visible)
        self.assertEqual(output[1].elem_classes, ["btn-glow-single"])
        self.assertFalse(output[2].visible)
        self.assertFalse(output[4].interactive)

    def test_unknown_language_shows_completado_badge(self):
        result = SimpleNamespace(
            formatted_text="Texto",
            detected_language="unknown",
            output_path=Path("audio_TRANSCRIPCION.txt"),
        )
        pipeline = Mock(process=Mock(return_value=result))
        with patch("app.create_transcription_pipeline", return_value=pipeline):
            output = app.process_transcription(
                "audio.wav",
                ProviderType.DEEPGRAM.value,
                "nova-2",
                False,
                "",
                deepgram_key_input="personal",
                progress=Mock(),
            )
        self.assertEqual(len(output), 5)
        self.assertIn("Completado", output[3])
        self.assertNotIn("UNKNOWN", output[3])
        self.assertFalse(output[2].visible)
        self.assertFalse(output[4].interactive)

    def test_whisperx_execution_in_app(self):
        result = SimpleNamespace(
            formatted_text="Hola desde WhisperX",
            detected_language="es",
            output_path=Path("audio_TRANSCRIPCION.txt"),
        )
        pipeline = Mock(process=Mock(return_value=result))
        with (
            patch("app.create_transcription_pipeline", return_value=pipeline),
            patch("app.execute_whisperx", return_value=result) as mock_exec,
        ):
            output = app.process_transcription(
                "audio.wav",
                ProviderType.WHISPERX.value,
                "large-v3-turbo",
                False,
                "",
                progress=Mock(),
            )
        mock_exec.assert_called_once()
        self.assertEqual(len(output), 5)
        self.assertEqual(output[0], "Hola desde WhisperX")
        self.assertIn("ES", output[3])
        self.assertTrue(output[1].visible)
        self.assertEqual(output[1].elem_classes, ["btn-glow-single"])
        self.assertFalse(output[2].visible)
        self.assertFalse(output[4].interactive)

    def test_login_success_failure_and_logout(self):
        with patch("app.verify_admin_pin", return_value=True):
            result = app.process_admin_login("pin", ProviderType.DEEPGRAM.value, False)
        self.assertTrue(result[0])
        self.assertEqual(result[-1], "")
        with patch("app.verify_admin_pin", return_value=False):
            result = app.process_admin_login("wrong", ProviderType.DEEPGRAM.value, False)
        self.assertFalse(result[0])
        self.assertEqual(result[-1], "")
        self.assertFalse(app.process_admin_logout(ProviderType.DEEPGRAM.value, False)[0])

    def test_provider_switch_and_guest_panels(self):
        for provider in ProviderType:
            with self.subTest(provider=provider):
                result = app.on_provider_selected(provider.value, False, False)
                self.assertEqual(len(result), 7)
                self.assertTrue(result[0]["choices"])
                self.assertEqual(len(app.update_guest_api_panel(provider.value, True, True)), 5)

    def test_clear_resets_media_text_download_and_status(self):
        result = app.clear_all()
        self.assertIsNone(result[0])
        self.assertEqual(result[1], "")
        self.assertFalse(result[2].interactive)
        self.assertEqual(result[2].elem_classes, ["btn-secondary"])
        self.assertFalse(result[3].visible)
        self.assertFalse(result[3].interactive)
        self.assertEqual(len(result), 6)
        self.assertTrue(result[5].interactive)

    def test_batch_transcription_multiple_files_creates_zip(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out1 = tmp_path / "audio1_TRANSCRIPCION.txt"
            out2 = tmp_path / "audio2_TRANSCRIPCION.txt"
            out1.write_text("Texto 1", encoding="utf-8")
            out2.write_text("Texto 2", encoding="utf-8")

            result1 = SimpleNamespace(
                formatted_text="Texto 1",
                detected_language="es",
                output_path=out1,
            )
            result2 = SimpleNamespace(
                formatted_text="Texto 2",
                detected_language="en",
                output_path=out2,
            )
            pipeline = Mock()
            pipeline.process.side_effect = [result1, result2]

            with (
                patch("app.create_transcription_pipeline", return_value=pipeline),
                patch("app.create_zip_archive", return_value=tmp_path / "lote.zip") as mock_zip,
            ):
                output = app.process_transcription(
                    ["audio1.wav", "audio2.wav"],
                    ProviderType.DEEPGRAM.value,
                    "nova-2",
                    False,
                    "",
                    deepgram_key_input="personal",
                    progress=Mock(),
                )

            self.assertEqual(len(output), 5)
            self.assertEqual(pipeline.process.call_count, 2)
            self.assertIn("📁 audio1.wav [Idioma: ES]", output[0])
            self.assertIn("📁 audio2.wav [Idioma: EN]", output[0])
            self.assertIn("Texto 1", output[0])
            self.assertIn("Texto 2", output[0])
            mock_zip.assert_called_once()
            self.assertTrue(output[1].interactive)
            self.assertTrue(output[1].visible)
            self.assertEqual(output[1].elem_classes, ["btn-glow-zip"])
            val = output[1].value
            val_path = val.get("path", "") if isinstance(val, dict) else str(val)
            self.assertTrue(val_path.endswith(".zip"))

            # Validar que el botón de descarga consolidada se active exclusivamente en lote
            self.assertTrue(output[2].interactive)
            self.assertTrue(output[2].visible)
            self.assertEqual(output[2].elem_classes, ["btn-glow-txt"])
            # Colores visualmente diferentes para descarga de zip y txt consolidado
            self.assertNotEqual(output[1].elem_classes, output[2].elem_classes)
            cons_val = output[2].value
            cons_path = cons_val.get("path", "") if isinstance(cons_val, dict) else str(cons_val)
            self.assertTrue(cons_path.endswith(".txt"))
            self.assertIn("transcripcion_consolidada_lote_", cons_path)

            self.assertIn("2 archivos procesados", output[3])
            # Botón Iniciar Transcripción deshabilitado tras finalizar
            self.assertFalse(output[4].interactive)

    def test_batch_transcription_partial_failure_continues(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            out2 = tmp_path / "audio2_TRANSCRIPCION.txt"
            out2.write_text("Texto 2", encoding="utf-8")

            result2 = SimpleNamespace(
                formatted_text="Texto 2",
                detected_language="es",
                output_path=out2,
            )
            pipeline = Mock()
            pipeline.process.side_effect = [RuntimeError("Archivo corrupto"), result2]

            with (
                patch("app.create_transcription_pipeline", return_value=pipeline),
                patch("app.create_zip_archive", return_value=tmp_path / "lote.zip") as mock_zip,
            ):
                output = app.process_transcription(
                    ["bad.wav", "good.wav"],
                    ProviderType.DEEPGRAM.value,
                    "nova-2",
                    False,
                    "",
                    deepgram_key_input="personal",
                    progress=Mock(),
                )

            self.assertEqual(len(output), 5)
            self.assertIn("⚠️ ARCHIVOS NO PROCESADOS (ERRORES)", output[0])
            self.assertIn("• bad.wav: Archivo corrupto", output[0])
            self.assertIn("📁 good.wav [Idioma: ES]", output[0])
            self.assertIn("Texto 2", output[0])
            mock_zip.assert_called_once()
            self.assertTrue(output[1].interactive)
            self.assertTrue(output[1].visible)
            self.assertEqual(output[1].elem_classes, ["btn-glow-zip"])
            self.assertTrue(output[2].interactive)
            self.assertTrue(output[2].visible)
            self.assertEqual(output[2].elem_classes, ["btn-glow-txt"])
            self.assertIn("1/2 procesados", output[3])
            self.assertFalse(output[4].interactive)

    def test_single_file_transcription_hides_consolidated_button(self):
        result = SimpleNamespace(
            formatted_text="Texto individual",
            detected_language="es",
            output_path=Path("audio_solo_TRANSCRIPCION.txt"),
        )
        pipeline = Mock(process=Mock(return_value=result))
        with patch("app.create_transcription_pipeline", return_value=pipeline):
            output = app.process_transcription(
                ["audio_solo.wav"],
                ProviderType.DEEPGRAM.value,
                "nova-2",
                False,
                "",
                deepgram_key_input="personal",
                progress=Mock(),
            )
        self.assertEqual(len(output), 5)
        self.assertTrue(output[1].visible)
        self.assertTrue(output[1].interactive)
        self.assertEqual(output[1].elem_classes, ["btn-glow-single"])
        # El botón de lote consolidado DEBE estar oculto para un solo archivo
        self.assertFalse(output[2].visible)
        self.assertFalse(output[2].interactive)
        # El botón Iniciar Transcripción se deshabilita hasta Limpiar
        self.assertFalse(output[4].interactive)

    def test_transcribe_button_disabled_until_clear_clicked(self):
        result = SimpleNamespace(
            formatted_text="Texto",
            detected_language="es",
            output_path=Path("audio_TRANSCRIPCION.txt"),
        )
        pipeline = Mock(process=Mock(return_value=result))
        with patch("app.create_transcription_pipeline", return_value=pipeline):
            output = app.process_transcription(
                "audio.wav",
                ProviderType.DEEPGRAM.value,
                "nova-2",
                False,
                "",
                deepgram_key_input="personal",
                progress=Mock(),
            )
        # Transcripción completada: Iniciar Transcripción desactivado
        self.assertFalse(output[4].interactive)

        # Usuario presiona Limpiar: Iniciar Transcripción reactivado
        clear_res = app.clear_all()
        self.assertTrue(clear_res[5].interactive)


if __name__ == "__main__":
    unittest.main()
