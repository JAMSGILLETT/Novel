"""
Prompts tab — view and edit every AI prompt used by the pipeline, in
pipeline order, plus the user-provided style sample used for prose
consistency.

Layout:
  Left panel  — Style Sample entry, then every prompt grouped by node
  Right panel — editor for the selected prompt (or the style sample)

Editing a prompt saves an override to the story_metadata/prompt_overrides
tables (see db.py, prompt_templates.py); "Reset to Default" removes the
override so the built-in wording is used again. Only the static
instructional wording is editable here — the per-chapter data blocks
(character lists, plotlines, etc.) are still assembled in code and slotted
into the template at its placeholders.
"""
from __future__ import annotations

from pathlib import Path

import customtkinter as ctk

from prompt_templates import PROMPT_REGISTRY, DEFAULT_TEMPLATES, get_template, save_template, reset_template

_STYLE_SAMPLE_KEY = "__style_sample__"  # sentinel, not a real prompt_overrides row


class PromptsTab:
    def __init__(self, parent_tab, story_id: str, db_path: Path):
        self.story_id = story_id
        self.db_path = db_path

        self._selected_key: str | None = None
        self._item_buttons: dict[str, ctk.CTkButton] = {}

        self._build(parent_tab)
        self._select(_STYLE_SAMPLE_KEY)

    # ── Layout ────────────────────────────────────────────────────────────

    def _build(self, tab):
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=0)
        tab.grid_columnconfigure(1, weight=1)

        # ── Left panel: grouped prompt list ────────────────────────────
        left = ctk.CTkFrame(tab, width=260)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(0, weight=1)
        left.grid_propagate(False)

        self._list_frame = ctk.CTkScrollableFrame(left, label_text="Prompts (pipeline order)")
        self._list_frame.grid(row=0, column=0, sticky="nsew", padx=4, pady=8)

        style_btn = ctk.CTkButton(
            self._list_frame,
            text="✦ Style Sample",
            anchor="w",
            fg_color=("gray75", "gray30"),
            hover_color=("gray70", "gray35"),
            font=("", 12, "bold"),
            command=lambda: self._select(_STYLE_SAMPLE_KEY),
        )
        style_btn.pack(fill="x", pady=(0, 8))
        self._item_buttons[_STYLE_SAMPLE_KEY] = style_btn

        last_group = None
        for group, key, description in PROMPT_REGISTRY:
            if group != last_group:
                lbl = ctk.CTkLabel(
                    self._list_frame, text=group, anchor="w",
                    font=("", 11, "bold"), text_color="gray60",
                )
                lbl.pack(fill="x", pady=(10, 2))
                last_group = group

            btn = ctk.CTkButton(
                self._list_frame,
                text=description,
                anchor="w",
                fg_color="transparent",
                hover_color=("gray75", "gray30"),
                font=("", 12),
                command=lambda k=key: self._select(k),
            )
            btn.pack(fill="x", pady=1)
            self._item_buttons[key] = btn

        # ── Right panel: editor ─────────────────────────────────────────
        right = ctk.CTkFrame(tab)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._title_label = ctk.CTkLabel(
            right, text="", font=("", 15, "bold"), anchor="w",
        )
        self._title_label.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))

        self._editor = ctk.CTkTextbox(right, font=("Consolas", 12), wrap="word")
        self._editor.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))

        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        self._save_btn = ctk.CTkButton(btn_row, text="Save", width=100, command=self._save)
        self._save_btn.pack(side="left", padx=(0, 8))

        self._reset_btn = ctk.CTkButton(
            btn_row, text="Reset to Default", width=140, command=self._reset,
            fg_color=("gray70", "gray30"), hover_color=("gray60", "gray40"),
        )
        self._reset_btn.pack(side="left")

        self._status_label = ctk.CTkLabel(btn_row, text="", font=("", 11))
        self._status_label.pack(side="left", padx=12)

    # ── Selection / loading ──────────────────────────────────────────────

    def _select(self, key: str):
        self._selected_key = key
        self._status_label.configure(text="")

        for k, btn in self._item_buttons.items():
            is_selected = (k == key)
            btn.configure(fg_color=("gray65", "gray25") if is_selected else (
                ("gray75", "gray30") if k == _STYLE_SAMPLE_KEY else "transparent"
            ))

        if key == _STYLE_SAMPLE_KEY:
            self._title_label.configure(text="Style Sample (prose reference for the Story Writer)")
            self._reset_btn.configure(state="disabled")
            import db
            current = db.get_style_sample(self.story_id, self.db_path) or ""
            self._set_editor_text(current)
        else:
            description = next((d for g, k, d in PROMPT_REGISTRY if k == key), key)
            self._title_label.configure(text=description)
            self._reset_btn.configure(state="normal")
            current = get_template(key, self.story_id, self.db_path)
            self._set_editor_text(current)

    def _set_editor_text(self, text: str):
        self._editor.configure(state="normal")
        self._editor.delete("1.0", "end")
        self._editor.insert("1.0", text)

    # ── Save / Reset ──────────────────────────────────────────────────────

    def _save(self):
        text = self._editor.get("1.0", "end").rstrip("\n")
        key = self._selected_key
        try:
            if key == _STYLE_SAMPLE_KEY:
                import db
                db.upsert_style_sample(self.story_id, text, self.db_path)
            else:
                save_template(key, self.story_id, text, self.db_path)
            self._status_label.configure(text="Saved.", text_color="#4caf50")
        except Exception as e:
            self._status_label.configure(text=f"Save failed: {e}", text_color="#f5a623")

    def _reset(self):
        key = self._selected_key
        if key == _STYLE_SAMPLE_KEY:
            return
        try:
            reset_template(key, self.story_id, self.db_path)
            self._set_editor_text(DEFAULT_TEMPLATES[key])
            self._status_label.configure(text="Reset to default.", text_color="#4caf50")
        except Exception as e:
            self._status_label.configure(text=f"Reset failed: {e}", text_color="#f5a623")
