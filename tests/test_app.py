"""Regresiones de eventos web y aislamiento de credenciales por sesión."""

import asyncio
import inspect
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from gradio.blocks import SessionState

import app
from src.auth import LoginAttemptLimiter
from src.config import ProviderType
from src.ui.theme import get_blocks_and_launch_kwargs, get_custom_theme


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

    def test_batch_graceful_stop_finishes_current_file_yields_partial_zip_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            files = []
            outputs = []
            for idx in (1, 2, 3):
                media = tmp_path / f"clase{idx}.wav"
                media.write_bytes(f"audio {idx}".encode())
                out = tmp_path / f"clase{idx}_TRANSCRIPCION.txt"
                out.write_text(f"Texto {idx}", encoding="utf-8")
                files.append(str(media))
                outputs.append(
                    SimpleNamespace(
                        formatted_text=f"Texto {idx}",
                        detected_language="es",
                        output_path=out,
                    )
                )

            req = SimpleNamespace(session_hash="sess_batch_pause")
            pipe_pause = Mock()

            def run_and_stop(media_path, progress_callback=None):
                stop_btn = app.toggle_graceful_stop(req)
                self.assertIn("Cancelar pausa", stop_btn.value)
                return outputs[0]

            pipe_pause.process.side_effect = run_and_stop

            with (
                patch("app.create_transcription_pipeline", return_value=pipe_pause),
                patch("app.create_zip_archive", return_value=tmp_path / "parcial.zip") as mock_zip,
            ):
                yields_pause = list(
                    app.process_transcription_stream(
                        files,
                        ProviderType.DEEPGRAM.value,
                        "nova-2",
                        False,
                        "",
                        deepgram_key_input="personal",
                        progress=Mock(),
                        request=req,
                    )
                )

            self.assertEqual(pipe_pause.process.call_count, 1)
            self.assertGreaterEqual(len(yields_pause), 2)
            first_yield = yields_pause[0]
            self.assertFalse(first_yield[4].visible)  # btn_transcribe oculto durante lote
            self.assertTrue(first_yield[5].visible)   # btn_stop_transcribe visible

            paused_state = yields_pause[-1]
            self.assertIn("Texto 1", paused_state[0])
            self.assertNotIn("Texto 2", paused_state[0])
            mock_zip.assert_called_once()
            self.assertTrue(paused_state[1].interactive)
            self.assertTrue(paused_state[2].interactive)
            self.assertIn("Pausado: 1/3 listos (2 pendientes)", paused_state[3])
            self.assertTrue(paused_state[4].visible)
            self.assertTrue(paused_state[4].interactive)
            self.assertEqual(paused_state[4].value, "▶️ Reanudar (2 pendientes)")
            self.assertFalse(paused_state[5].visible)
            paused_cache = paused_state[6]
            self.assertTrue(paused_cache["paused"])

            # Al reevaluar cambios en UI con misma configuración mantiene etiqueta de reanudación
            btn_ui = app.on_media_files_changed(
                files,
                paused_cache,
                provider=ProviderType.DEEPGRAM.value,
                model_name="nova-2",
                enable_diarization=False,
                timestamp_option="",
            )
            self.assertEqual(btn_ui.value, "▶️ Reanudar (2 pendientes)")

            # Reanudar desde donde se quedó: solo debe procesar clase2 y clase3
            pipe_resume = Mock()
            pipe_resume.process.side_effect = [outputs[1], outputs[2]]
            with (
                patch("app.create_transcription_pipeline", return_value=pipe_resume),
                patch("app.create_zip_archive", return_value=tmp_path / "completo.zip"),
            ):
                yields_resume = list(
                    app.process_transcription_stream(
                        files,
                        ProviderType.DEEPGRAM.value,
                        "nova-2",
                        False,
                        "",
                        deepgram_key_input="personal",
                        batch_cache=paused_cache,
                        progress=Mock(),
                        request=req,
                    )
                )

            self.assertEqual(pipe_resume.process.call_count, 2)
            final_state = yields_resume[-1]
            self.assertIn("Texto 1", final_state[0])
            self.assertIn("Texto 2", final_state[0])
            self.assertIn("Texto 3", final_state[0])
            self.assertIn("3 listos (2 nuevos, 1 en caché)", final_state[3])
            self.assertFalse(final_state[4].interactive)
            self.assertFalse(final_state[5].visible)
            self.assertFalse(final_state[6]["paused"])

    def test_graceful_stop_can_be_toggled_off_before_file_finishes(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            files = []
            outputs = []
            for idx in (1, 2):
                media = tmp_path / f"v{idx}.wav"
                media.write_bytes(f"audio {idx}".encode())
                out = tmp_path / f"v{idx}_TRANSCRIPCION.txt"
                out.write_text(f"Txt {idx}", encoding="utf-8")
                files.append(str(media))
                outputs.append(
                    SimpleNamespace(
                        formatted_text=f"Txt {idx}",
                        detected_language="es",
                        output_path=out,
                    )
                )

            req = SimpleNamespace(session_hash="sess_undo")
            pipe = Mock()

            def run_toggle_twice(media_path, progress_callback=None):
                b1 = app.toggle_graceful_stop(req)
                self.assertIn("Cancelar pausa", b1.value)
                b2 = app.toggle_graceful_stop(req)
                self.assertEqual(b2.value, "🛑 Detener tras este archivo")
                return outputs[0]

            # Simular side_effect con callable que despacha por llamada
            call_num = {"n": 0}

            def dispatch(media_path, progress_callback=None):
                call_num["n"] += 1
                if call_num["n"] == 1:
                    return run_toggle_twice(media_path, progress_callback)
                return outputs[1]

            pipe.process.side_effect = dispatch

            with (
                patch("app.create_transcription_pipeline", return_value=pipe),
                patch("app.create_zip_archive", return_value=tmp_path / "all.zip"),
            ):
                yields = list(
                    app.process_transcription_stream(
                        files,
                        ProviderType.DEEPGRAM.value,
                        "nova-2",
                        False,
                        "",
                        deepgram_key_input="personal",
                        progress=Mock(),
                        request=req,
                    )
                )

            self.assertEqual(call_num["n"], 2)
            self.assertIn("2 archivos procesados", yields[-1][3])
            self.assertFalse(yields[-1][6]["paused"])

    def test_single_file_does_not_show_stop_button(self):
        result = SimpleNamespace(
            formatted_text="Uno solo",
            detected_language="es",
            output_path=Path("uno_TRANSCRIPCION.txt"),
        )
        pipeline = Mock(process=Mock(return_value=result))
        with patch("app.create_transcription_pipeline", return_value=pipeline):
            yields = list(
                app.process_transcription_stream(
                    ["uno.wav"],
                    ProviderType.DEEPGRAM.value,
                    "nova-2",
                    False,
                    "",
                    deepgram_key_input="personal",
                    progress=Mock(),
                )
            )
        self.assertEqual(len(yields), 1)
        self.assertFalse(yields[0][5].visible)

    def test_config_change_invalidates_cached_results_on_resume(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            f1 = tmp_path / "a1.wav"
            f2 = tmp_path / "a2.wav"
            f1.write_bytes(b"111")
            f2.write_bytes(b"222")
            out1 = tmp_path / "a1_TRANSCRIPCION.txt"
            out2 = tmp_path / "a2_TRANSCRIPCION.txt"
            out1.write_text("Old 1", encoding="utf-8")
            out2.write_text("New 2", encoding="utf-8")

            _, fp1 = app._get_file_fingerprint(str(f1))
            paused_cache = {
                "cached_results": {
                    fp1: ("a1.wav", SimpleNamespace(formatted_text="Old 1", detected_language="es", output_path=out1))
                },
                "config_signature": app.build_config_signature(
                    ProviderType.DEEPGRAM.value, "nova-2", False, "simple"
                ),
                "paused": True,
            }

            # Al cambiar de proveedor, la UI vuelve a "Iniciar Transcripción" en lugar de "Reanudar"
            btn_changed = app.on_media_files_changed(
                [str(f1), str(f2)],
                paused_cache,
                provider=ProviderType.ASSEMBLYAI.value,
                model_name="assemblyai-stt-3-7",
                enable_diarization=False,
                timestamp_option="Simple ([HH:MM:SS])",
            )
            self.assertEqual(btn_changed.value, "⚡ Iniciar Transcripción")

            pipe_new = Mock()
            pipe_new.process.side_effect = [
                SimpleNamespace(formatted_text="New 1", detected_language="es", output_path=out1),
                SimpleNamespace(formatted_text="New 2", detected_language="es", output_path=out2),
            ]
            with (
                patch("app.create_transcription_pipeline", return_value=pipe_new),
                patch("app.create_zip_archive"),
            ):
                output = app.process_transcription(
                    [str(f1), str(f2)],
                    ProviderType.ASSEMBLYAI.value,
                    "assemblyai-stt-3-7",
                    False,
                    "Simple ([HH:MM:SS])",
                    assemblyai_key_input="personal",
                    batch_cache=paused_cache,
                    progress=Mock(),
                )

            self.assertEqual(pipe_new.process.call_count, 2)
            self.assertIn("New 1", output[0])
            self.assertNotIn("Old 1", output[0])

    def test_session_isolation_for_graceful_stop(self):
        req_a = SimpleNamespace(session_hash="session_a")
        req_b = SimpleNamespace(session_hash="session_b")
        try:
            # Sin lote activo, toggle_graceful_stop ignora la solicitud y oculta el botón
            idle_btn = app.toggle_graceful_stop(req_a)
            self.assertFalse(idle_btn.visible)
            self.assertFalse(app._is_stop_requested(app._get_session_key(req_a)))

            # Con lote activo en session_a, solo afecta a session_a y no a session_b
            app._set_batch_active("session_a", "run_a")
            active_btn = app.toggle_graceful_stop(req_a)
            self.assertTrue(active_btn.visible)
            self.assertTrue(app._is_stop_requested(app._get_session_key(req_a)))
            self.assertFalse(app._is_stop_requested(app._get_session_key(req_b)))
        finally:
            app._set_batch_active("session_a", None)
            app._set_batch_active("session_b", None)
            app._set_stop_requested("session_a", False)
            app._set_stop_requested("session_b", False)

    def test_clear_all_ui_and_empty_change_keep_transcribe_enabled(self):
        req = SimpleNamespace(session_hash="sess_clear_ui")
        try:
            app._set_batch_active("sess_clear_ui", "run_clear")
            app._set_stop_requested("sess_clear_ui", True)
            res = app.clear_all_ui(req)
            self.assertEqual(len(res), 8)
            self.assertIsNone(res[0])
            self.assertEqual(res[1], "")
            self.assertTrue(res[5].interactive)
            self.assertEqual(res[6], {"cached_results": {}})
            self.assertFalse(res[7].visible)
            self.assertFalse(app._is_stop_requested("sess_clear_ui"))
            self.assertNotIn("sess_clear_ui", app._ACTIVE_BATCH_SESSIONS)

            stale_cache = {
                "cached_results": {"a.wav_10": ("a.wav", SimpleNamespace(output_path=Path("a.txt")))}
            }
            btn = app._refresh_transcribe_button_on_ui_change(None, stale_cache)
            self.assertTrue(btn.interactive)
            self.assertEqual(btn.value, "⚡ Iniciar Transcripción")
        finally:
            app._set_batch_active("sess_clear_ui", None)
            app._set_stop_requested("sess_clear_ui", False)

    def test_generator_close_aborts_cleanly_and_clears_session_state(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            files = []
            for idx in (1, 2):
                media = tmp_path / f"abort{idx}.wav"
                media.write_bytes(f"audio {idx}".encode())
                files.append(str(media))

            req = SimpleNamespace(session_hash="sess_gen_abort")
            pipeline = Mock()
            with (
                patch("app.create_transcription_pipeline", return_value=pipeline),
                patch("app.clear_gpu_vram") as mock_clear_vram,
            ):
                gen = app.process_transcription_stream(
                    files,
                    ProviderType.DEEPGRAM.value,
                    "nova-2",
                    False,
                    "",
                    deepgram_key_input="personal",
                    progress=Mock(),
                    request=req,
                )
                first_yield = next(gen)
                self.assertFalse(first_yield[4].visible)
                self.assertTrue(first_yield[5].visible)
                self.assertIn("sess_gen_abort", app._ACTIVE_BATCH_SESSIONS)
                app.toggle_graceful_stop(req)
                self.assertTrue(app._is_stop_requested("sess_gen_abort"))
                gen.close()

            self.assertNotIn("sess_gen_abort", app._ACTIVE_BATCH_SESSIONS)
            self.assertFalse(app._is_stop_requested("sess_gen_abort"))
            self.assertTrue(mock_clear_vram.called)

    def test_batch_download_failure_mid_stream_returns_controlled_error(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            files = []
            outputs = []
            for idx in (1, 2):
                media = tmp_path / f"zipfail{idx}.wav"
                media.write_bytes(f"audio {idx}".encode())
                out = tmp_path / f"zipfail{idx}_TRANSCRIPCION.txt"
                out.write_text(f"Texto {idx}", encoding="utf-8")
                files.append(str(media))
                outputs.append(
                    SimpleNamespace(
                        formatted_text=f"Texto {idx}",
                        detected_language="es",
                        output_path=out,
                    )
                )

            pipeline = Mock()
            pipeline.process.side_effect = outputs
            with (
                patch("app.create_transcription_pipeline", return_value=pipeline),
                patch("app.create_zip_archive", side_effect=PermissionError("disco lleno")),
            ):
                yields = list(
                    app.process_transcription_stream(
                        files,
                        ProviderType.DEEPGRAM.value,
                        "nova-2",
                        False,
                        "",
                        deepgram_key_input="personal",
                        progress=Mock(),
                    )
                )

            final_state = yields[-1]
            self.assertEqual(len(final_state), 7)
            self.assertIn("error inesperado", final_state[0].lower())
            self.assertNotIn("disco lleno", final_state[0])
            self.assertFalse(final_state[1].interactive)
            self.assertFalse(final_state[2].visible)
            self.assertTrue(final_state[4].interactive)
            self.assertFalse(final_state[5].visible)
            self.assertEqual(len(final_state[6]["cached_results"]), 2)

    def test_update_copy_button_disables_for_errors_and_preserves_gradio_state(self):
        for non_copyable in [
            "",
            "   ",
            "⚠️ Por favor selecciona o arrastra al menos un archivo.",
            "❌ Error de transcripción:\nClave inválida",
            "⏸️ Lote pausado. Presiona Reanudar para continuar.",
        ]:
            with self.subTest(text=non_copyable):
                upd = app._update_copy_button(non_copyable)
                self.assertFalse(upd["interactive"])

        upd_ok = app._update_copy_button("Hola mundo transcrito")
        self.assertTrue(upd_ok["interactive"])
        upd_partial = app._update_copy_button(
            f"{'=' * 10}\n📁 a.wav\n{'=' * 10}\nTexto\n\n⚠️ ARCHIVOS NO PROCESADOS"
        )
        self.assertTrue(upd_partial["interactive"])

        # Verificar que el delta enviado a Gradio solo actualiza interactive sin mutar elem_classes ni enviar 'Run'
        copy_fn = next(
            fn for fn in app.demo.fns.values() if fn.fn is app._update_copy_button
        )
        state = SessionState(app.demo)
        res = asyncio.run(
            app.demo.process_api(block_fn=copy_fn, inputs=["Texto válido"], state=state)
        )
        self.assertEqual(res["data"], [{"interactive": True, "__type__": "update"}])
        self.assertEqual(state[app.ui.btn_copy._id].elem_classes, ["btn-secondary"])

    def test_admin_session_token_prevents_state_bypass_and_enforces_ttl(self):
        req_a = SimpleNamespace(session_hash="admin_sess_a", client=SimpleNamespace(host="198.51.100.10"))
        req_b = SimpleNamespace(session_hash="admin_sess_b", client=SimpleNamespace(host="198.51.100.11"))
        try:
            with patch.dict(
                "os.environ",
                {
                    "DEEPGRAM_API_KEY": "server-dg",
                    "ASSEMBLYAI_API_KEY": "server-aai",
                    "HF_TOKEN": "server-hf",
                },
            ):
                # Bypass con True forjado o token falso rechazado cuando hay request HTTP
                for forged in (True, "forged_token", ""):
                    _, dg_key, dg_err = app._resolve_provider_credentials(
                        ProviderType.DEEPGRAM.value, False, "", "", "", forged, request=req_a
                    )
                    self.assertIsNone(dg_key)
                    self.assertIsNotNone(dg_err)

                    hf_tok, _, hf_err = app._resolve_provider_credentials(
                        ProviderType.WHISPERX.value, True, "", "", "", forged, request=req_a
                    )
                    self.assertIsNone(hf_tok)
                    self.assertIsNotNone(hf_err)

                # Login válido emite token ligado a req_a
                with patch("app.verify_admin_pin", return_value=True):
                    login_res = app.process_admin_login("1234", ProviderType.DEEPGRAM.value, False, req_a)
                token_a = login_res[0]
                self.assertIsInstance(token_a, str)
                self.assertGreater(len(token_a), 16)

                # Token de req_a funciona en req_a (para Deepgram y HF_TOKEN) y falla en req_b
                _, dg_ok, dg_err_ok = app._resolve_provider_credentials(
                    ProviderType.DEEPGRAM.value, False, "", "", "", token_a, request=req_a
                )
                self.assertEqual(dg_ok, "server-dg")
                self.assertIsNone(dg_err_ok)

                hf_ok, _, hf_err_ok = app._resolve_provider_credentials(
                    ProviderType.WHISPERX.value, True, "", "", "", token_a, request=req_a
                )
                self.assertEqual(hf_ok, "server-hf")
                self.assertIsNone(hf_err_ok)

                _, dg_cross, dg_cross_err = app._resolve_provider_credentials(
                    ProviderType.DEEPGRAM.value, False, "", "", "", token_a, request=req_b
                )
                self.assertIsNone(dg_cross)
                self.assertIsNotNone(dg_cross_err)

                # TTL expirado invalida la sesión
                with patch("app.time.monotonic", return_value=time.monotonic() + 7200):
                    _, dg_exp, dg_exp_err = app._resolve_provider_credentials(
                        ProviderType.DEEPGRAM.value, False, "", "", "", token_a, request=req_a
                    )
                self.assertIsNone(dg_exp)
                self.assertIsNotNone(dg_exp_err)

                # Re-login + logout revoca inmediatamente
                with patch("app.verify_admin_pin", return_value=True):
                    token_a2 = app.process_admin_login("1234", ProviderType.DEEPGRAM.value, False, req_a)[0]
                app.process_admin_logout(ProviderType.DEEPGRAM.value, False, req_a)
                _, dg_rev, dg_rev_err = app._resolve_provider_credentials(
                    ProviderType.DEEPGRAM.value, False, "", "", "", token_a2, request=req_a
                )
                self.assertIsNone(dg_rev)
                self.assertIsNotNone(dg_rev_err)

                # ALLOW_PUBLIC_HF_TOKEN="true" habilita HF_TOKEN del servidor para invitados
                with patch.dict("os.environ", {"ALLOW_PUBLIC_HF_TOKEN": "true"}):
                    hf_pub, _, hf_pub_err = app._resolve_provider_credentials(
                        ProviderType.WHISPERX.value, True, "", "", "", False, request=req_b
                    )
                self.assertEqual(hf_pub, "server-hf")
                self.assertIsNone(hf_pub_err)
        finally:
            app._revoke_admin_session(req_a)
            app._revoke_admin_session(req_b)

    def test_web_media_path_rejects_paths_outside_allowed_roots(self):
        real_pipeline = SimpleNamespace(process=lambda **kwargs: None)
        outside_file = Path(__file__).resolve()
        with self.assertRaises(app.AudioProcessingError):
            app._validate_web_media_path(str(outside_file), real_pipeline, False)

    def test_gradio_event_arities_and_launch_security_config(self):
        for fn_obj in app.demo.fns.values():
            if fn_obj.fn is None:
                continue
            sig = inspect.signature(fn_obj.fn)
            non_special = [
                p
                for name, p in sig.parameters.items()
                if name not in ("request", "progress")
            ]
            required = [
                p for p in non_special if p.default is inspect.Parameter.empty
            ]
            self.assertGreaterEqual(len(fn_obj.inputs), len(required))
            self.assertLessEqual(len(fn_obj.inputs), len(non_special))
            if fn_obj.fn is not app.process_transcription_stream:
                self.assertEqual(fn_obj.api_visibility, "private")

        _, launch_cfg = get_blocks_and_launch_kwargs("", get_custom_theme(), "")
        self.assertEqual(launch_cfg["max_file_size"], "2gb")
        self.assertIn(".env", launch_cfg["blocked_paths"])
        with patch.dict("os.environ", {"MAX_UPLOAD_SIZE_MB": "512"}):
            _, custom_launch = get_blocks_and_launch_kwargs("", get_custom_theme(), "")
        self.assertEqual(custom_launch["max_file_size"], "512mb")

        # Verificar que no se duplica <style> en gr.HTML dentro del árbol de Blocks
        html_values = [
            str(getattr(block, "value", "") or "")
            for block in app.demo.blocks.values()
            if isinstance(block, app.gr.HTML)
        ]
        self.assertFalse(any("<style>" in val for val in html_values))

    def test_ui_css_and_js_safety_contracts(self):
        css_text = Path("assets/css/theme.css").read_text(encoding="utf-8")
        js_text = Path("assets/js/ui_enhancements.js").read_text(encoding="utf-8")

        for selector in (
            ".btn-login-compact.hidden",
            "#btn_transcribe.hidden",
            "#btn_stop_transcribe.hidden",
            ".result-card .actions-row .hidden",
            ".result-card .actions-row button:not(.hidden):not(.hide)",
            "#btn_transcribe.btn-transcribe-queued[data-queue-label]::after",
            "@media (min-width: 1025px) and (max-height: 559px)",
        ):
            with self.subTest(selector=selector):
                self.assertIn(selector, css_text)

        self.assertNotIn("innerHTML", js_text)
        self.assertNotIn("onclick=", js_text)
        self.assertNotIn("btn.textContent =", js_text)
        self.assertIn("data-queue-label", js_text)


if __name__ == "__main__":
    unittest.main()
