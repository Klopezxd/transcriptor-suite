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
        self.assertEqual(len(result), 6)
        self.assertFalse(result[1].interactive)
        self.assertFalse(result[2].visible)
        self.assertTrue(result[4].interactive)
        self.assertEqual(result[5], {"cached_results": {}})

    def test_unknown_provider_returns_controlled_error(self):
        result = app.process_transcription("audio.wav", "unknown", "model", False, "")
        self.assertEqual(len(result), 6)
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
        self.assertEqual(len(output), 6)
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
        self.assertEqual(len(output), 6)
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
        self.assertEqual(len(output), 6)
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
        self.assertEqual(len(result), 7)
        self.assertTrue(result[5].interactive)
        self.assertEqual(result[6], {"cached_results": {}})

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

            self.assertEqual(len(output), 6)
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
            self.assertIn("cached_results", output[5])
            self.assertEqual(len(output[5]["cached_results"]), 2)

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

            self.assertEqual(len(output), 6)
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
        self.assertEqual(len(output), 6)
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
        self.assertEqual(clear_res[6], {"cached_results": {}})

    def test_differential_batch_transcription_skips_cached_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            file1 = tmp_path / "clase1.mp3"
            file2 = tmp_path / "clase2.mp3"
            file1.write_bytes(b"dummy audio 1")
            file2.write_bytes(b"dummy audio 2")

            out1 = tmp_path / "clase1_TRANSCRIPCION.txt"
            out2 = tmp_path / "clase2_TRANSCRIPCION.txt"
            out1.write_text("Transcripcion clase 1", encoding="utf-8")
            out2.write_text("Transcripcion clase 2", encoding="utf-8")

            res1 = SimpleNamespace(
                formatted_text="Transcripcion clase 1",
                detected_language="es",
                output_path=out1,
            )
            res2 = SimpleNamespace(
                formatted_text="Transcripcion clase 2",
                detected_language="es",
                output_path=out2,
            )

            # Paso 1: Transcribir solo file1
            pipe1 = Mock()
            pipe1.process.return_value = res1
            with (
                patch("app.create_transcription_pipeline", return_value=pipe1),
                patch("app.create_zip_archive"),
            ):
                output1 = app.process_transcription(
                    [str(file1)],
                    ProviderType.DEEPGRAM.value,
                    "nova-2",
                    False,
                    "",
                    deepgram_key_input="personal",
                    batch_cache=None,
                    progress=Mock(),
                )

            session_cache = output1[5]
            self.assertEqual(pipe1.process.call_count, 1)
            self.assertEqual(len(session_cache["cached_results"]), 1)

            # Paso 2: Usuario añade file2 (file1 + file2) pasando el session_cache
            pipe2 = Mock()
            pipe2.process.return_value = res2
            with (
                patch("app.create_transcription_pipeline", return_value=pipe2),
                patch("app.create_zip_archive"),
            ):
                output2 = app.process_transcription(
                    [str(file1), str(file2)],
                    ProviderType.DEEPGRAM.value,
                    "nova-2",
                    False,
                    "",
                    deepgram_key_input="personal",
                    batch_cache=session_cache,
                    progress=Mock(),
                )

            # Solo se debió procesar file2, file1 provino de caché
            self.assertEqual(pipe2.process.call_count, 1)
            self.assertIn("Transcripcion clase 1", output2[0])
            self.assertIn("Transcripcion clase 2", output2[0])
            # Badge refleja 1 nuevo y 1 en caché
            self.assertIn("1 nuevos, 1 en caché", output2[3])
            # Ambos archivos quedan en el caché actualizado
            updated_cache = output2[5]
            self.assertEqual(len(updated_cache["cached_results"]), 2)

    def test_on_media_files_changed_reactivates_when_new_file_added(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = tmp_path / "audio1.wav"
            f2 = tmp_path / "audio2.wav"
            f3 = tmp_path / "audio3.wav"
            f1.write_bytes(b"audio1")
            f2.write_bytes(b"audio2")
            f3.write_bytes(b"audio3")

            out1 = tmp_path / "audio1_TRANSCRIPCION.txt"
            out1.write_text("Transcripcion 1", encoding="utf-8")

            # Sin archivos seleccionados y sin caché -> Botón habilitado estándar
            btn = app.on_media_files_changed(None, None)
            self.assertTrue(btn.interactive)
            self.assertEqual(btn.value, "⚡ Iniciar Transcripción")

            # 1 archivo y sin caché previo -> Botón activo estándar
            btn = app.on_media_files_changed([str(f1)], None)
            self.assertTrue(btn.interactive)
            self.assertEqual(btn.value, "⚡ Iniciar Transcripción")

            # Simular caché con f1 ya procesado y archivo en disco
            _, fp1 = app._get_file_fingerprint(str(f1))
            mock_cache = {
                "cached_results": {
                    fp1: (
                        f1.name,
                        SimpleNamespace(output_path=out1),
                    )
                }
            }
            # Si solo está f1 y ya está en caché -> Botón desactivado
            btn = app.on_media_files_changed([str(f1)], mock_cache)
            self.assertFalse(btn.interactive)

            # Si se quitan los archivos pero persiste caché de sesión -> Botón desactivado
            btn_empty = app.on_media_files_changed(None, mock_cache)
            self.assertFalse(btn_empty.interactive)

            # Si se agrega f2 -> Botón se reactiva indicando 1 archivo nuevo
            btn = app.on_media_files_changed([str(f1), str(f2)], mock_cache)
            self.assertTrue(btn.interactive)
            self.assertEqual(btn.value, "⚡ Transcribir 1 archivo nuevo")

            # Si se agregan f2 y f3 -> Botón indica 2 archivos nuevos
            btn = app.on_media_files_changed([str(f1), str(f2), str(f3)], mock_cache)
            self.assertTrue(btn.interactive)
            self.assertEqual(btn.value, "⚡ Transcribir 2 archivos nuevos")


if __name__ == "__main__":
    unittest.main()
