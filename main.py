"""
NovelGen — GUI entry point.

Requires:  pip install customtkinter
Package:   pyinstaller --onefile --windowed main.py
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import customtkinter as ctk

# ---------------------------------------------------------------------------
# Story config — edit these or expose via a settings screen later
# ---------------------------------------------------------------------------
STORY_ID    = "my-story"
BOOK_TITLE  = "My Novel"
DB_PATH     = Path(__file__).parent / "story.db"
CHROMA_PATH = Path(__file__).parent / "story_chroma"
MANUSCRIPTS = Path(__file__).parent / "manuscripts"


# ---------------------------------------------------------------------------
# Queue-backed print writer
# ---------------------------------------------------------------------------
class _QueueWriter:
    def __init__(self, q: queue.Queue):
        self._q = q

    def __call__(self, text: str) -> None:
        self._q.put(("log", text + "\n"))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class NovelGenApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("NovelGen")
        self.geometry("960x680")
        self.minsize(700, 450)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self._output_queue: queue.Queue = queue.Queue()
        self._running = False

        self._build_ui()
        self._poll_queue()
        self._refresh_chapter_list()

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._tabs = ctk.CTkTabview(self, anchor="nw")
        self._tabs.grid(row=0, column=0, sticky="nsew", padx=10, pady=(10, 0))

        self._build_chapters_tab()
        self._build_chat_tab()
        self._build_world_bible_tab()
        self._build_prompts_tab()

        self._status = ctk.CTkLabel(
            self, text="Ready.", anchor="w",
            text_color="gray60", font=("", 11),
        )
        self._status.grid(row=1, column=0, sticky="ew", padx=12, pady=(2, 6))

    # ── Chapters tab ──────────────────────────────────────────────────────

    def _build_chapters_tab(self):
        tab = self._tabs.add("Chapters")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=0)  # list panel — fixed width
        tab.grid_columnconfigure(1, weight=1)  # prose panel — fills remaining

        # Left: scrollable chapter list
        self._chapter_list_frame = ctk.CTkScrollableFrame(
            tab, width=160, label_text=BOOK_TITLE, label_font=("", 12, "bold"),
        )
        self._chapter_list_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        self._chapter_buttons: list[ctk.CTkButton] = []

        # Right: prose viewer
        self._prose_view = ctk.CTkTextbox(
            tab,
            state="disabled",
            font=("Georgia", 13),
            wrap="word",
        )
        self._prose_view.grid(row=0, column=1, sticky="nsew")

    def _refresh_chapter_list(self):
        """Rebuild the chapter list from disk."""
        for btn in self._chapter_buttons:
            btn.destroy()
        self._chapter_buttons.clear()

        book_dir = MANUSCRIPTS / BOOK_TITLE
        if not book_dir.exists():
            return

        chapters = sorted(
            book_dir.glob("Chapter *.txt"),
            key=lambda p: int(p.stem.split()[-1]),
        )
        for path in chapters:
            label = path.stem  # "Chapter 1", "Chapter 2", …
            btn = ctk.CTkButton(
                self._chapter_list_frame,
                text=label,
                anchor="w",
                fg_color="transparent",
                hover_color=("gray75", "gray30"),
                font=("", 12),
                command=lambda p=path: self._load_chapter(p),
            )
            btn.pack(fill="x", pady=2)
            self._chapter_buttons.append(btn)

    def _load_chapter(self, path: Path):
        text = path.read_text(encoding="utf-8")
        self._prose_view.configure(state="normal")
        self._prose_view.delete("1.0", "end")
        self._prose_view.insert("end", text)
        self._prose_view.configure(state="disabled")
        self._prose_view.see("1.0")

    # ── Chat tab ──────────────────────────────────────────────────────────

    def _build_world_bible_tab(self):
        from world_bible_tab import WorldBibleTab
        tab = self._tabs.add("World Bible")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        WorldBibleTab(tab, STORY_ID, DB_PATH, CHROMA_PATH)

    def _build_prompts_tab(self):
        from prompts_tab import PromptsTab
        tab = self._tabs.add("Prompts")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)
        PromptsTab(tab, STORY_ID, DB_PATH)

    def _build_chat_tab(self):
        tab = self._tabs.add("Chat")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=1)

        # Streaming log output
        self._chat_output = ctk.CTkTextbox(
            tab,
            state="disabled",
            font=("Consolas", 12),
            wrap="word",
        )
        self._chat_output.grid(row=0, column=0, columnspan=2,
                                sticky="nsew", pady=(0, 8))

        self._chat_output._textbox.tag_config("header",  foreground="#5bc8f5")
        self._chat_output._textbox.tag_config("prose",   foreground="#e0e0e0")
        self._chat_output._textbox.tag_config("warning", foreground="#f5a623")
        self._chat_output._textbox.tag_config("dim",     foreground="#888888")

        # Input row
        tab.grid_rowconfigure(1, weight=0)
        self._chat_input = ctk.CTkEntry(
            tab,
            placeholder_text="What happens next?",
            font=("", 13),
            height=38,
        )
        self._chat_input.grid(row=1, column=0, sticky="ew", pady=(0, 0))
        self._chat_input.bind("<Return>", lambda e: self._send_chapter())

        self._send_btn = ctk.CTkButton(
            tab, text="Generate", width=110, height=38,
            command=self._send_chapter,
        )
        self._send_btn.grid(row=1, column=1, padx=(8, 0))

    # ── Chapter generation ────────────────────────────────────────────────

    def _send_chapter(self):
        if self._running:
            return
        user_input = self._chat_input.get().strip()
        if not user_input:
            return

        self._chat_input.delete(0, "end")
        self._set_busy(True)
        self._append_log(f"\n>>> {user_input}\n", tag="header")

        writer = _QueueWriter(self._output_queue)
        threading.Thread(
            target=self._run_pipeline,
            args=(user_input, writer),
            daemon=True,
        ).start()

    def _run_pipeline(self, user_input: str, writer: _QueueWriter):
        try:
            from pipeline import run_chapter
            state = run_chapter(
                user_input=user_input,
                story_id=STORY_ID,
                db_path=DB_PATH,
                chroma_path=CHROMA_PATH,
                manuscripts_dir=MANUSCRIPTS,
                book_title=BOOK_TITLE,
                print_fn=writer,
            )
            # Signal GUI to refresh chapter list
            self._output_queue.put(("refresh_chapters", None))

        except Exception as exc:
            import traceback
            writer(f"\n[ERROR] {exc}")
            writer(traceback.format_exc())
        finally:
            self._output_queue.put(("done", None))

    # ── Output streaming ──────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._output_queue.get_nowait()
                if kind == "log":
                    tag = self._classify_tag(payload)
                    self._append_log(payload, tag=tag)
                elif kind == "refresh_chapters":
                    self._refresh_chapter_list()
                elif kind == "done":
                    self._set_busy(False)
        except queue.Empty:
            pass
        self.after(50, self._poll_queue)

    def _classify_tag(self, text: str) -> str:
        t = text.strip()
        if t.startswith("NODE ") or t.startswith("===") or t.startswith(">>>"):
            return "header"
        if t.startswith("[warn]") or t.startswith("[ERROR]") or "FLAGGED" in t:
            return "warning"
        if t.startswith("Done in") or t.startswith("─") or t.startswith("Saved:"):
            return "dim"
        return "prose"

    def _append_log(self, text: str, tag: str = "prose"):
        box = self._chat_output
        box.configure(state="normal")
        box._textbox.insert("end", text, tag)
        box.see("end")
        box.configure(state="disabled")

    # ── Helpers ───────────────────────────────────────────────────────────

    def _set_busy(self, busy: bool):
        self._running = busy
        self._send_btn.configure(state="disabled" if busy else "normal")
        self._chat_input.configure(state="disabled" if busy else "normal")
        self._status.configure(
            text="Generating…  (this may take a few minutes)" if busy else "Ready."
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = NovelGenApp()
    app.mainloop()
