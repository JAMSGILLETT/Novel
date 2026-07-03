"""
NovelGen — GUI entry point.

Requires:  pip install customtkinter
Package:   pyinstaller --onefile --windowed main.py
"""
from __future__ import annotations

import queue
import re
import threading
import time
from pathlib import Path
from tkinter import messagebox

import customtkinter as ctk

import db
import llm_client

# ---------------------------------------------------------------------------
# Paths (per-install, not per-story). Story identity is chosen in the GUI.
# ---------------------------------------------------------------------------
DB_PATH     = Path(__file__).parent / "story.db"
CHROMA_PATH = Path(__file__).parent / "story_chroma"
MANUSCRIPTS = Path(__file__).parent / "manuscripts"

DEFAULT_STORY_ID   = "my-story"
DEFAULT_BOOK_TITLE = "My Novel"

_WARN_COLOR = "#f5a623"

_FLAG_RE  = re.compile(r"FLAGGED FOR REVIEW")
_WORDS_RE = re.compile(r"\((\d+)\s+words\)")


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "story"


# ---------------------------------------------------------------------------
# Queue-backed print writer
# ---------------------------------------------------------------------------
class _QueueWriter:
    def __init__(self, q: queue.Queue):
        self._q = q

    def __call__(self, text: str) -> None:
        self._q.put(("log", text + "\n"))


def _fmt_elapsed(secs: float) -> str:
    return f"{secs:.0f}s" if secs < 60 else f"{int(secs // 60)}m {int(secs % 60)}s"


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
class NovelGenApp(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("NovelGen")
        self.geometry("960x680")
        self.minsize(760, 480)
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Ensure the DB + default story exist, then adopt the first story.
        db.init_db(DB_PATH)
        if db.get_story_title(DEFAULT_STORY_ID, DB_PATH) is None:
            db.create_story(DEFAULT_STORY_ID, DEFAULT_BOOK_TITLE, DB_PATH)
        stories = db.list_stories(DB_PATH)
        self._story_id   = stories[0]["story_id"]
        self._book_title = stories[0]["book_title"]

        # Adopt the saved model choice (Settings tab), if any.
        saved_model = db.get_setting("model", None, DB_PATH)
        if saved_model:
            llm_client.set_model(saved_model)

        self._output_queue: queue.Queue = queue.Queue()
        self._running = False
        self._cancel_event = threading.Event()
        self._gen_start = 0.0
        self._stage_label = ""
        self._stage_step = 0
        self._stage_total = 0

        self._build_ui()
        self._poll_queue()
        self._refresh_chapter_list()
        self._notify_pending_resume()

    def _notify_pending_resume(self):
        """If a crash left an interrupted chapter behind, tell the user how to
        pick it up — otherwise they'd only discover it on their next Send."""
        try:
            ckpt = db.get_chapter_checkpoint(self._story_id, DB_PATH)
        except Exception:
            return
        if ckpt is None:
            return
        self._append_log(
            f"⟳ Chapter {ckpt['chapter_number']} was interrupted (last completed node: "
            f"{ckpt['last_stage']}). Type anything and press Send to resume it from there.\n",
            tag="warning",
        )
        self._status.configure(text=f"Interrupted chapter {ckpt['chapter_number']} found — send a message to resume.")

    # ── UI construction ───────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self._build_story_bar()

        self._tabs = ctk.CTkTabview(self, anchor="nw")
        self._tabs.grid(row=1, column=0, sticky="nsew", padx=10, pady=(6, 0))

        self._build_chapters_tab()
        self._build_chat_tab()
        self._build_world_bible_tab()
        self._build_prompts_tab()
        self._build_settings_tab()

        # Bottom bar: status label (left) + debug checkbox (right)
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.grid(row=2, column=0, sticky="ew", padx=12, pady=(2, 6))
        bottom.grid_columnconfigure(0, weight=1)

        self._status = ctk.CTkLabel(
            bottom, text="Ready.", anchor="w",
            text_color="gray60", font=("", 11),
        )
        self._status.grid(row=0, column=0, sticky="w")

        saved_debug = db.get_setting("debug_mode", "0", DB_PATH) == "1"
        self._debug_var = ctk.BooleanVar(value=saved_debug)
        self._debug_var.trace_add("write", lambda *_: db.set_setting("debug_mode", "1" if self._debug_var.get() else "0", DB_PATH))
        ctk.CTkCheckBox(
            bottom,
            text="Debug mode",
            variable=self._debug_var,
            font=("", 11),
            checkbox_width=16,
            checkbox_height=16,
        ).grid(row=0, column=1, sticky="e", padx=(8, 0))

    # ── Story bar (switcher + new) ────────────────────────────────────────

    def _build_story_bar(self):
        bar = ctk.CTkFrame(self, fg_color="transparent")
        bar.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 0))

        ctk.CTkLabel(bar, text="Story:", font=("", 12)).pack(side="left", padx=(0, 6))

        self._story_menu = ctk.CTkOptionMenu(
            bar, values=self._story_titles(), width=220,
            command=self._on_story_selected,
        )
        self._story_menu.set(self._book_title)
        self._story_menu.pack(side="left")

        ctk.CTkButton(
            bar, text="+ New", width=70, command=self._new_story,
        ).pack(side="left", padx=(8, 0))

    def _story_titles(self) -> list[str]:
        self._story_options = {s["book_title"]: s["story_id"] for s in db.list_stories(DB_PATH)}
        return list(self._story_options.keys())

    def _on_story_selected(self, title: str):
        if self._running:
            self._story_menu.set(self._book_title)  # ignore switch mid-run
            return
        story_id = self._story_options.get(title)
        if not story_id or story_id == self._story_id:
            return
        self._story_id = story_id
        self._book_title = title
        self._load_story_tabs()
        self._clear_prose()
        self._refresh_chapter_list()
        self._status.configure(text=f"Switched to “{title}”.")
        self._notify_pending_resume()

    def _new_story(self):
        if self._running:
            return
        dialog = ctk.CTkInputDialog(text="Title for the new story:", title="New story")
        title = (dialog.get_input() or "").strip()
        if not title:
            return
        # Derive a unique story_id from the title.
        base = _slugify(title)
        existing = {s["story_id"] for s in db.list_stories(DB_PATH)}
        story_id, n = base, 2
        while story_id in existing:
            story_id, n = f"{base}-{n}", n + 1

        db.create_story(story_id, title, DB_PATH)
        self._story_menu.configure(values=self._story_titles())
        self._story_menu.set(title)
        self._on_story_selected(title)

    # ── Chapters tab ──────────────────────────────────────────────────────

    def _build_chapters_tab(self):
        tab = self._tabs.add("Chapters")
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=0)  # list panel — fixed width
        tab.grid_columnconfigure(1, weight=1)  # prose panel — fills remaining

        # Left: scrollable chapter list
        self._chapter_list_frame = ctk.CTkScrollableFrame(
            tab, width=200, label_text=self._book_title, label_font=("", 12, "bold"),
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

    def _chapter_meta(self, path: Path) -> tuple[bool, int]:
        """Return (flagged, word_count) parsed from a chapter file's header,
        falling back to counting the body if the header lacks a word count."""
        try:
            text = path.read_text(encoding="utf-8")
        except OSError:
            return False, 0
        head = text[:400]
        flagged = bool(_FLAG_RE.search(head))
        m = _WORDS_RE.search(head)
        if m:
            return flagged, int(m.group(1))
        # Fallback: count body words (skip the header block up to the blank line)
        body = text.split("\n\n", 1)[-1]
        return flagged, len(body.split())

    def _refresh_chapter_list(self):
        """Rebuild the chapter list from disk, with flag markers + word counts."""
        for btn in self._chapter_buttons:
            btn.destroy()
        self._chapter_buttons.clear()

        book_dir = MANUSCRIPTS / self._book_title
        chapters = (
            sorted(book_dir.glob("Chapter *.txt"), key=lambda p: int(p.stem.split()[-1]))
            if book_dir.exists() else []
        )

        total_words = 0
        for path in chapters:
            flagged, wc = self._chapter_meta(path)
            total_words += wc
            n = path.stem.split()[-1]
            label = f"{'⚠ ' if flagged else ''}Chapter {n}  ·  {wc:,}w"
            btn = ctk.CTkButton(
                self._chapter_list_frame,
                text=label,
                anchor="w",
                fg_color="transparent",
                text_color=_WARN_COLOR if flagged else None,
                hover_color=("gray75", "gray30"),
                font=("", 12),
                command=lambda p=path: self._load_chapter(p),
            )
            btn.pack(fill="x", pady=2)
            self._chapter_buttons.append(btn)

        summary = (
            f"{self._book_title}  —  {len(chapters)} ch · {total_words:,}w"
            if chapters else self._book_title
        )
        self._chapter_list_frame.configure(label_text=summary)

    def _load_chapter(self, path: Path):
        text = path.read_text(encoding="utf-8")
        self._prose_view.configure(state="normal")
        self._prose_view.delete("1.0", "end")
        self._prose_view.insert("end", text)
        self._prose_view.configure(state="disabled")
        self._prose_view.see("1.0")

    def _clear_prose(self):
        self._prose_view.configure(state="normal")
        self._prose_view.delete("1.0", "end")
        self._prose_view.configure(state="disabled")

    # ── World Bible / Prompts tabs ────────────────────────────────────────

    def _build_world_bible_tab(self):
        self._wb_tab = self._tabs.add("World Bible")
        self._wb_tab.grid_rowconfigure(0, weight=1)
        self._wb_tab.grid_columnconfigure(0, weight=1)
        self._prompts_tab = self._tabs.add("Prompts")
        self._prompts_tab.grid_rowconfigure(0, weight=1)
        self._prompts_tab.grid_columnconfigure(0, weight=1)
        self._load_story_tabs()

    def _build_prompts_tab(self):
        # World Bible + Prompts are built together in _build_world_bible_tab so
        # they can be rebuilt as a pair when the active story changes.
        pass

    def _load_story_tabs(self):
        """(Re)populate the World Bible and Prompts tabs for the current story."""
        from world_bible_tab import WorldBibleTab
        from prompts_tab import PromptsTab
        for child in self._wb_tab.winfo_children():
            child.destroy()
        WorldBibleTab(self._wb_tab, self._story_id, DB_PATH, CHROMA_PATH)
        for child in self._prompts_tab.winfo_children():
            child.destroy()
        PromptsTab(self._prompts_tab, self._story_id, DB_PATH)

    # ── Settings tab ──────────────────────────────────────────────────────

    _MODEL_SUGGESTIONS = [
        "qwen2.5:14b-instruct-q4_K_M",
        "qwen2.5:7b-instruct",
        "llama3.1:8b",
        "llama3.3:70b",
        "mistral-nemo",
    ]

    def _build_settings_tab(self):
        tab = self._tabs.add("Settings")
        tab.grid_columnconfigure(0, weight=1)

        frame = ctk.CTkFrame(tab, fg_color="transparent")
        frame.grid(row=0, column=0, sticky="new", padx=20, pady=20)

        ctk.CTkLabel(frame, text="Generation model", font=("", 15, "bold")).pack(anchor="w")
        ctk.CTkLabel(
            frame,
            text=("The Ollama model used for every generation step. Pick from the list or type any "
                  "model you've pulled, then Save. Takes effect on your next chapter."),
            text_color="gray60", font=("", 11), justify="left", wraplength=560,
        ).pack(anchor="w", pady=(2, 12))

        self._model_box = ctk.CTkComboBox(frame, width=380, values=self._model_choices())
        self._model_box.set(llm_client.get_model())
        self._model_box.pack(anchor="w")

        row = ctk.CTkFrame(frame, fg_color="transparent")
        row.pack(anchor="w", pady=(12, 0))
        ctk.CTkButton(row, text="Save", width=90, command=self._save_model).pack(side="left")
        ctk.CTkButton(
            row, text="Detect installed", width=150,
            fg_color="gray30", hover_color="gray40", command=self._detect_models,
        ).pack(side="left", padx=(8, 0))

        self._settings_status = ctk.CTkLabel(
            frame, text=f"Current model: {llm_client.get_model()}",
            text_color="gray60", font=("", 11),
        )
        self._settings_status.pack(anchor="w", pady=(12, 0))

        # ── Generation quality ───────────────────────────────────────────
        ctk.CTkLabel(frame, text="Generation quality", font=("", 15, "bold")).pack(anchor="w", pady=(26, 0))
        ctk.CTkLabel(
            frame,
            text=("Craft self-refine: before the canon/craft check, the model does a quick pass to "
                  "improve its own draft's writing (pacing, tension, show-don't-tell). This usually "
                  "means fewer expensive revision rounds and stronger prose, at the cost of one or "
                  "two extra model calls per chapter. Continuity is never touched by this step. "
                  "On by default; takes effect on your next chapter."),
            text_color="gray60", font=("", 11), justify="left", wraplength=560,
        ).pack(anchor="w", pady=(2, 8))

        saved_refine = db.get_setting("self_refine", "1", DB_PATH) != "0"
        self._self_refine_var = ctk.BooleanVar(value=saved_refine)
        self._self_refine_var.trace_add(
            "write",
            lambda *_: db.set_setting("self_refine", "1" if self._self_refine_var.get() else "0", DB_PATH),
        )
        ctk.CTkCheckBox(
            frame, text="Enable craft self-refine", variable=self._self_refine_var,
            font=("", 12), checkbox_width=18, checkbox_height=18,
        ).pack(anchor="w")

        # ── Backups ──────────────────────────────────────────────────────
        ctk.CTkLabel(frame, text="Backups", font=("", 15, "bold")).pack(anchor="w", pady=(26, 0))
        ctk.CTkLabel(
            frame,
            text=("Full snapshots of the database — which covers every story — taken automatically "
                  "before each chapter. Restoring rolls the entire database back to that snapshot; "
                  "your current state is saved first, so a restore can itself be undone."),
            text_color="gray60", font=("", 11), justify="left", wraplength=560,
        ).pack(anchor="w", pady=(2, 8))

        brow = ctk.CTkFrame(frame, fg_color="transparent")
        brow.pack(anchor="w")
        ctk.CTkButton(brow, text="Back up now", width=120, command=self._backup_now).pack(side="left")
        ctk.CTkButton(
            brow, text="Refresh", width=90,
            fg_color="gray30", hover_color="gray40", command=self._refresh_backups,
        ).pack(side="left", padx=(8, 0))

        self._backups_frame = ctk.CTkScrollableFrame(
            frame, width=560, height=190, label_text="Available snapshots (newest first)",
        )
        self._backups_frame.pack(anchor="w", fill="x", pady=(8, 0))
        self._backup_rows: list = []
        self._refresh_backups()

    # ── Backups ───────────────────────────────────────────────────────────

    def _refresh_backups(self):
        for w in self._backup_rows:
            w.destroy()
        self._backup_rows.clear()

        backups = db.list_backups(DB_PATH)
        if not backups:
            empty = ctk.CTkLabel(self._backups_frame, text="No backups yet.",
                                 text_color="gray60", font=("", 11), anchor="w")
            empty.pack(anchor="w", pady=4)
            self._backup_rows.append(empty)
            return

        for b in backups:
            r = ctk.CTkFrame(self._backups_frame, fg_color="transparent")
            r.pack(fill="x", pady=2)
            ctk.CTkLabel(
                r, text=f"{b['label']}  ·  {b['when']}  ·  {b['size_kb']:,} KB",
                font=("", 11), anchor="w",
            ).pack(side="left")
            ctk.CTkButton(
                r, text="Restore", width=80,
                command=lambda p=b["path"], lbl=b["label"], when=b["when"]: self._restore_backup(p, lbl, when),
            ).pack(side="right")
            self._backup_rows.append(r)

    def _backup_now(self):
        if self._running:
            self._settings_status.configure(text="Can't back up while generating.", text_color=_WARN_COLOR)
            return
        try:
            dest = db.create_backup(DB_PATH, tag="manual")
        except Exception as e:
            self._settings_status.configure(text=f"Backup failed: {e}", text_color=_WARN_COLOR)
            return
        self._refresh_backups()
        self._settings_status.configure(
            text=f"Backed up: {dest.name}" if dest else "Nothing to back up yet.",
            text_color="#4caf50" if dest else _WARN_COLOR,
        )

    def _restore_backup(self, path: Path, label: str, when: str):
        if self._running:
            self._settings_status.configure(text="Can't restore while generating.", text_color=_WARN_COLOR)
            return
        if not messagebox.askyesno(
            "Restore backup",
            f"Restore “{label}” from {when}?\n\n"
            "This rolls the ENTIRE database (all stories) back to that snapshot. "
            "Your current state will be saved first as a pre-restore backup, so this can be undone.",
            icon="warning", default="no",
        ):
            return
        try:
            db.restore_backup(path, DB_PATH)
        except Exception as e:
            self._settings_status.configure(text=f"Restore failed: {e}", text_color=_WARN_COLOR)
            return
        self._reload_after_restore()
        self._settings_status.configure(
            text=f"Restored “{label}”. Previous state saved as a pre-restore backup.",
            text_color="#4caf50",
        )

    def _reload_after_restore(self):
        """Re-adopt state from the freshly restored DB and refresh the whole UI."""
        db.init_db(DB_PATH)  # ensure schema exists if an older DB was restored
        stories = db.list_stories(DB_PATH)
        if not stories:
            db.create_story(DEFAULT_STORY_ID, DEFAULT_BOOK_TITLE, DB_PATH)
            stories = db.list_stories(DB_PATH)

        ids = {s["story_id"]: s["book_title"] for s in stories}
        if self._story_id not in ids:
            self._story_id = stories[0]["story_id"]
            self._book_title = stories[0]["book_title"]
        else:
            self._book_title = ids[self._story_id]

        saved_model = db.get_setting("model", None, DB_PATH)
        if saved_model:
            llm_client.set_model(saved_model)

        self._story_menu.configure(values=self._story_titles())
        self._story_menu.set(self._book_title)
        self._model_box.set(llm_client.get_model())
        self._load_story_tabs()
        self._clear_prose()
        self._refresh_chapter_list()
        self._refresh_backups()

    def _model_choices(self) -> list[str]:
        current = llm_client.get_model()
        choices = list(self._MODEL_SUGGESTIONS)
        if current not in choices:
            choices.insert(0, current)
        return choices

    def _save_model(self):
        name = (self._model_box.get() or "").strip()
        if not name:
            return
        llm_client.set_model(name)
        db.set_setting("model", llm_client.get_model(), DB_PATH)
        self._settings_status.configure(
            text=f"Saved — generating with: {llm_client.get_model()}", text_color="#4caf50",
        )

    def _detect_models(self):
        self._settings_status.configure(text="Querying Ollama…", text_color="gray60")
        self.update_idletasks()
        try:
            models = llm_client.list_installed_models()
        except Exception as e:
            self._settings_status.configure(
                text=f"Couldn't reach Ollama ({e}). Is it running?", text_color=_WARN_COLOR,
            )
            return
        if not models:
            self._settings_status.configure(
                text="Ollama reports no installed models. Pull one with `ollama pull …`.",
                text_color=_WARN_COLOR,
            )
            return
        self._model_box.configure(values=models)
        self._settings_status.configure(
            text=f"Found {len(models)} installed model(s) — pick one and Save.", text_color="gray60",
        )

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
        self._chat_output.grid(row=0, column=0, columnspan=3,
                               sticky="nsew", pady=(0, 8))

        self._chat_output._textbox.tag_config("header",  foreground="#5bc8f5")
        self._chat_output._textbox.tag_config("prose",   foreground="#e0e0e0")
        self._chat_output._textbox.tag_config("warning", foreground="#f5a623")
        self._chat_output._textbox.tag_config("dim",     foreground="#888888")

        # Input row — multi-line textbox: Enter = newline, Shift+Enter = send
        tab.grid_rowconfigure(1, weight=0)
        self._chat_input = ctk.CTkTextbox(
            tab,
            font=("", 13),
            height=70,
            wrap="word",
        )
        self._chat_input.grid(row=1, column=0, sticky="ew", pady=(0, 0))
        self._chat_input.bind("<Shift-Return>", lambda e: (self._send_chapter(), "break")[1])
        self._chat_input.bind("<Return>", lambda e: None)  # let Enter insert newline normally

        self._send_btn = ctk.CTkButton(
            tab, text="Generate", width=110, height=38,
            command=self._send_chapter,
        )
        self._send_btn.grid(row=1, column=1, padx=(8, 0))

        self._stop_btn = ctk.CTkButton(
            tab, text="Stop", width=70, height=38,
            fg_color="#8a3b3b", hover_color="#a94747",
            state="disabled", command=self._cancel_generation,
        )
        self._stop_btn.grid(row=1, column=2, padx=(8, 0))

    # ── Chapter generation ────────────────────────────────────────────────

    def _send_chapter(self):
        if self._running:
            return
        user_input = self._chat_input.get("1.0", "end").strip()
        if not user_input:
            return

        self._chat_input.delete("1.0", "end")
        self._cancel_event.clear()
        self._gen_start = time.time()
        self._stage_label, self._stage_step, self._stage_total = "Starting…", 0, 0
        self._set_busy(True)
        self._append_log(f"\n>>> {user_input}\n", tag="header")

        writer = _QueueWriter(self._output_queue)
        threading.Thread(
            target=self._run_pipeline,
            args=(user_input, writer),
            daemon=True,
        ).start()

    def _cancel_generation(self):
        if not self._running:
            return
        self._cancel_event.set()
        self._stop_btn.configure(state="disabled")
        self._append_log("\n⏹ Stopping after the current step…\n", tag="warning")

    def _run_pipeline(self, user_input: str, writer: _QueueWriter):
        try:
            from pipeline import run_chapter, GenerationCancelled

            def progress(step: int, total: int, label: str):
                self._output_queue.put(("progress", (step, total, label)))

            try:
                run_chapter(
                    user_input=user_input,
                    story_id=self._story_id,
                    db_path=DB_PATH,
                    chroma_path=CHROMA_PATH,
                    manuscripts_dir=MANUSCRIPTS,
                    book_title=self._book_title,
                    print_fn=writer,
                    should_cancel=self._cancel_event.is_set,
                    progress_fn=progress,
                    debug=self._debug_var.get(),
                )
                self._output_queue.put(("refresh_chapters", None))
            except GenerationCancelled:
                writer("\n⏹ Generation cancelled — no chapter saved.")

        except Exception as exc:
            import traceback
            writer(f"\n[ERROR] {exc}")
            writer(traceback.format_exc())
            try:
                import db as db_module
                if db_module.get_chapter_checkpoint(self._story_id, DB_PATH) is not None:
                    writer(
                        "\n⟳ Progress was checkpointed — type anything and press Send "
                        "to resume this chapter from the last completed node."
                    )
            except Exception:
                pass
        finally:
            self._output_queue.put(("done", None))

    # ── Output streaming ──────────────────────────────────────────────────

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self._output_queue.get_nowait()
                if kind == "log":
                    self._append_log(payload, tag=self._classify_tag(payload))
                elif kind == "progress":
                    self._stage_step, self._stage_total, self._stage_label = payload
                elif kind == "refresh_chapters":
                    self._refresh_chapter_list()
                elif kind == "done":
                    self._set_busy(False)
        except queue.Empty:
            pass

        # While running, keep the status line's elapsed clock live.
        if self._running:
            self._update_running_status()
        self.after(100, self._poll_queue)

    def _update_running_status(self):
        elapsed = _fmt_elapsed(time.time() - self._gen_start)
        prefix = "Cancelling — " if self._cancel_event.is_set() else ""
        if self._stage_total:
            self._status.configure(
                text=f"{prefix}{self._stage_label}  ·  stage {self._stage_step}/{self._stage_total}  ·  {elapsed}"
            )
        else:
            self._status.configure(text=f"{prefix}{self._stage_label}  ·  {elapsed}")

    def _classify_tag(self, text: str) -> str:
        t = text.strip()
        if t.startswith("NODE ") or t.startswith("===") or t.startswith(">>>"):
            return "header"
        if t.startswith("[warn]") or t.startswith("[ERROR]") or "FLAGGED" in t or t.startswith("⏹"):
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
        self._stop_btn.configure(state="normal" if busy else "disabled")
        self._chat_input.configure(state="disabled" if busy else "normal")
        self._story_menu.configure(state="disabled" if busy else "normal")
        if not busy:
            self._status.configure(text="Ready.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = NovelGenApp()
    app.mainloop()
