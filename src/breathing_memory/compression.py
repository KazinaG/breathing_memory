from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Callable, Optional, Protocol


@dataclass(frozen=True)
class CompressionResult:
    content: str
    target_length: int


class CompressionBackend(Protocol):
    def compress(self, content: str, compression_ratio: float) -> CompressionResult:
        ...


def _build_codex_compression_prompt(content: str, compression_ratio: float) -> str:
    retained_ratio = max(0.0, 1.0 - compression_ratio)
    retained_percent = max(1, int(round(retained_ratio * 100)))
    return (
        "You compress one memory fragment for Breathing Memory.\n"
        "Return only the compressed fragment text.\n"
        "Do not add commentary, bullets, quotes, or labels.\n"
        "Highest priority: minimize character count aggressively.\n"
        "Second priority: preserve the semantic core.\n"
        "Loss of peripheral detail is acceptable.\n"
        f"Target retained length: about {retained_percent}% of the original text.\n\n"
        "Fragment to compress:\n"
        f"{content}"
    )


class CodexExecCompressionBackend:
    def __init__(
        self,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        codex_path: Optional[str] = None,
        fallback: Optional[CompressionBackend] = None,
        workdir: Optional[Path] = None,
    ) -> None:
        self.runner = runner
        self.codex_path = codex_path
        self.fallback = fallback or StubCompressionBackend()
        self.workdir = workdir

    def compress(self, content: str, compression_ratio: float) -> CompressionResult:
        codex = self.codex_path or shutil.which("codex")
        if not codex:
            return self.fallback.compress(content, compression_ratio)

        prompt = _build_codex_compression_prompt(content, compression_ratio)
        target_length = max(1, int(round(len(" ".join(content.split())) * max(0.0, 1.0 - compression_ratio))))
        try:
            with tempfile.TemporaryDirectory(prefix="breathing-memory-compress-") as tempdir:
                output_path = Path(tempdir) / "compressed.txt"
                command = [
                    codex,
                    "exec",
                    "--ephemeral",
                    "--color",
                    "never",
                    "--output-last-message",
                    str(output_path),
                    "-",
                ]
                completed = self.runner(
                    command,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    cwd=str(self.workdir or Path.cwd()),
                    check=False,
                )
                if completed.returncode != 0 or not output_path.exists():
                    return self.fallback.compress(content, compression_ratio)
                compressed = output_path.read_text(encoding="utf-8").strip()
                if not compressed:
                    return self.fallback.compress(content, compression_ratio)
                return CompressionResult(content=compressed, target_length=target_length)
        except OSError:
            return self.fallback.compress(content, compression_ratio)


class StubCompressionBackend:
    def compress(self, content: str, compression_ratio: float) -> CompressionResult:
        normalized = " ".join(content.split())
        target_length = max(1, int(round(len(normalized) * max(0.0, 1.0 - compression_ratio))))
        if len(normalized) <= target_length:
            return CompressionResult(content=normalized, target_length=target_length)
        if target_length <= 3:
            compressed = normalized[:target_length]
        else:
            prefix = max(1, target_length - 3)
            compressed = normalized[:prefix] + "..."
        return CompressionResult(content=compressed, target_length=target_length)
