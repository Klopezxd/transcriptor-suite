"""Contrato de argumentos, códigos de salida y selección de archivos del CLI."""

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cli
from src.utils import TranscriptionError


class CliTests(unittest.TestCase):
    def test_argument_parsing(self):
        with patch("sys.argv", ["cli", "-i", "audio.wav", "--timestamps", "range", "-d"]):
            arguments = cli.parse_arguments()
        self.assertEqual(arguments.input, "audio.wav")
        self.assertEqual(arguments.timestamps, "range")
        self.assertTrue(arguments.diarize)

    def test_cancelled_selection_exits_successfully(self):
        with (
            patch("sys.argv", ["cli"]),
            patch("cli.select_file_gui", return_value=None),
            patch("builtins.print"),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main()
        self.assertEqual(error.exception.code, 0)

    def test_missing_file_exits_with_error(self):
        with (
            patch("sys.argv", ["cli", "-i", "missing-audit-file.wav"]),
            patch("builtins.print"),
            self.assertRaises(SystemExit) as error,
        ):
            cli.main()
        self.assertEqual(error.exception.code, 1)

    def test_success_and_pipeline_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            media = Path(directory) / "audio.wav"
            media.write_bytes(b"audio")
            for failure, code in [
                (None, None),
                (TranscriptionError("failure"), 1),
                (KeyboardInterrupt(), 130),
            ]:
                pipeline = Mock()
                pipeline.process.return_value = SimpleNamespace(
                    output_path=media.with_suffix(".txt")
                )
                pipeline.process.side_effect = failure
                with (
                    self.subTest(failure=failure),
                    patch("sys.argv", ["cli", "-i", str(media)]),
                    patch("cli.create_transcription_pipeline", return_value=pipeline),
                    patch("builtins.print"),
                ):
                    if code is None:
                        cli.main()
                    else:
                        with self.assertRaises(SystemExit) as error:
                            cli.main()
                        self.assertEqual(error.exception.code, code)

    def test_gui_window_is_destroyed_on_dialog_failure(self):
        root = Mock()
        tkinter = SimpleNamespace(
            Tk=Mock(return_value=root),
            filedialog=SimpleNamespace(
                askopenfilename=Mock(side_effect=RuntimeError("dialog failed"))
            ),
        )
        with patch.dict("sys.modules", {"tkinter": tkinter}):
            self.assertIsNone(cli.select_file_gui())
        root.destroy.assert_called_once()

    def test_provider_argument_aliases(self):
        for raw_provider, expected in [
            ("whisperx", "WhisperX (ZeroGPU / Local)"),
            ("deepgram", "Deepgram (Cloud API)"),
            ("assemblyai", "AssemblyAI (Cloud API)"),
            ("WhisperX (ZeroGPU / Local)", "WhisperX (ZeroGPU / Local)"),
        ]:
            with (
                self.subTest(provider=raw_provider),
                patch("sys.argv", ["cli", "-i", "audio.wav", "-p", raw_provider]),
            ):
                arguments = cli.parse_arguments()
                self.assertEqual(arguments.provider, expected)

    def test_invalid_provider_argument_fails(self):
        with (
            patch("sys.argv", ["cli", "-i", "audio.wav", "-p", "invalid_provider"]),
            patch("sys.stderr.write"),
            self.assertRaises(SystemExit),
        ):
            cli.parse_arguments()


if __name__ == "__main__":
    unittest.main()
