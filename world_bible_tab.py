"""
World Bible tab — view, add, and edit all story entities.

Layout:
  Left panel  — type dropdown + scrollable entity list
  Right panel — dynamic form for the selected / new entity
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import customtkinter as ctk

# ---------------------------------------------------------------------------
# Field spec format:
#   (field_name, label, widget_type, required)
#
# widget_type:
#   "entry"  — single-line CTkEntry
#   "text"   — multi-line CTkTextbox (longer content)
#   "list"   — CTkTextbox, one item per line
#   "dict"   — CTkTextbox, "Key: Value" one per line
#   "option" — CTkOptionMenu (pass choices as 5th element)
#   "int"    — CTkEntry, validated as integer
#   "bool"   — CTkOptionMenu fixed to True / False
# ---------------------------------------------------------------------------

FIELD_SPECS: dict[str, list] = {
    "Character": [
        ("name",                "Name",                              "entry",  True),
        ("personality",         "Personality",                       "text",   True),
        ("emotional_state",     "Emotional State",                   "entry",  False),
        ("is_alive",            "Alive",                             "bool",   False),
        ("goals",               "Goals  (one per line)",             "list",   False),
        ("current_objectives",  "Current Objectives  (one per line)","list",   False),
        ("knowledge",           "Knowledge  (one per line)",         "list",   False),
        ("secrets",             "Secrets  (one per line)",           "list",   False),
        ("relationships",       "Relationships  (Name: description, one per line)", "dict", False),
        ("current_location_id", "Current Location",                  "ref",    False, "location"),
    ],
    "Location": [
        ("name",              "Name",                             "entry",  True),
        ("description",       "Description",                      "text",   True),
        ("tone",              "Tone / Atmosphere",                "entry",  False),
        ("political_control", "Political Control",                "entry",  False),
        ("recent_events",     "Recent Events  (one per line)",    "list",   False),
        ("secrets",           "Secrets  (one per line)",          "list",   False),
        ("npcs_present",      "NPCs Present  (one per line)",     "list",   False),
    ],
    "Plotline": [
        ("name",           "Name",                                       "entry",  True),
        ("status",         "Status",                                     "option", True,
         ["active", "dormant", "resolved", "abandoned"]),
        ("progress_stage", "Progress Stage",                             "text",   True),
        ("current_tension","Tension  (0 – 10)",                         "int",    True),
        ("next_possible_developments", "Next Possible Developments  (one per line)", "list", False),
        ("involved_character_ids",     "Involved Characters",                        "ref_list", False, "character"),
    ],
    "World Rule": [
        ("rule_type", "Rule Type", "option", True,
         ["magic_system", "physics", "social_rule", "hard_constraint"]),
        ("title",   "Title",   "entry", True),
        ("content", "Content", "text",  True),
    ],
    "World Lore": [
        ("category", "Category", "option", True,
         ["history", "politics", "geography", "culture", "canon_fact"]),
        ("title",   "Title",   "entry", True),
        ("content", "Content", "text",  True),
    ],
    "POV State": [
        ("location_id",      "Location  (required)",             "ref",    True,  "location"),
        ("pov_character_id", "POV Character",                    "ref",    False, "character"),
        ("emotional_state",  "Emotional State",                  "entry",  False),
        ("companions",       "Companions",                       "ref_list", False, "character"),
        ("inventory",        "Inventory  (one per line)",        "list",   False),
        ("injuries",         "Injuries  (one per line)",         "list",   False),
        ("goals",            "Goals  (one per line)",            "list",   False),
        ("knowledge",        "Knowledge  (one per line)",        "list",   False),
    ],
    # One per story (like POV State). Beats and character arcs are managed by
    # the pipeline (mechanical updates + periodic LLM revision), not edited here.
    "Story Outline": [
        ("premise",        "Premise  (1–2 sentence core idea the story planner steers by)", "text", False),
        ("theme",          "Theme  (thematic throughline)",                                 "text", False),
        ("planned_ending", "Planned Ending  (rough direction — it will evolve)",           "text", False),
    ],
    # Dependency-graph rule: whenever the trigger entity is pulled into a chapter's
    # context, the inject entity is force-included too (bypassing relevance), with
    # the reason shown to the planner/writer as mandatory context.
    "Canon Rule": [
        ("trigger", "When this entity appears in a chapter…", "entity_any", True),
        ("inject",  "…always pull in this entity",            "entity_any", True),
        ("reason",  "Reason  (why they're linked — shown to the writer)", "text", True),
    ],
    # ── WorldEntity-backed types (category + name + description + attributes) ──
    "Organization": [
        ("name",        "Name",                          "entry", True),
        ("description", "Description",                   "text",  True),
        ("_attr_leader",       "Leader",                 "entry", False),
        ("_attr_members",      "Members  (one per line)","list",  False),
        ("_attr_goals",        "Goals  (one per line)",  "list",  False),
        ("_attr_headquarters", "Headquarters",           "entry", False),
        ("_attr_allies",       "Allies  (one per line)", "list",  False),
        ("_attr_enemies",      "Enemies  (one per line)","list",  False),
    ],
    "Culture": [
        ("name",        "Name",                                       "entry", True),
        ("description", "Description",                                "text",  True),
        ("_attr_values",        "Core Values  (one per line)",        "list",  False),
        ("_attr_traditions",    "Traditions  (one per line)",         "list",  False),
        ("_attr_taboos",        "Taboos  (one per line)",             "list",  False),
        ("_attr_language",      "Language",                           "entry", False),
        ("_attr_region",        "Region / Territory",                 "entry", False),
    ],
    "Species": [
        ("name",        "Name",                                       "entry", True),
        ("description", "Description",                                "text",  True),
        ("_attr_abilities",     "Abilities  (one per line)",          "list",  False),
        ("_attr_weaknesses",    "Weaknesses  (one per line)",         "list",  False),
        ("_attr_lifespan",      "Lifespan",                          "entry", False),
        ("_attr_habitat",       "Habitat",                            "entry", False),
        ("_attr_relations",     "Relations with other species  (one per line)", "list", False),
    ],
    "Skill": [
        ("name",        "Name",                                       "entry", True),
        ("description", "Description",                                "text",  True),
        ("_attr_category",      "Category  (e.g. combat, magic)",     "entry", False),
        ("_attr_prerequisites", "Prerequisites  (one per line)",      "list",  False),
        ("_attr_effects",       "Effects  (one per line)",            "list",  False),
        ("_attr_users",         "Known Users  (one per line)",        "list",  False),
    ],
    "Religion": [
        ("name",        "Name",                                       "entry", True),
        ("description", "Description",                                "text",  True),
        ("_attr_deity",         "Deity / Deities",                    "entry", False),
        ("_attr_tenets",        "Core Tenets  (one per line)",        "list",  False),
        ("_attr_rituals",       "Rituals  (one per line)",            "list",  False),
        ("_attr_holy_sites",    "Holy Sites  (one per line)",         "list",  False),
        ("_attr_followers",     "Follower Groups  (one per line)",    "list",  False),
    ],
    "Politics": [
        ("name",        "Name  (faction, government, etc.)",          "entry", True),
        ("description", "Description",                                "text",  True),
        ("_attr_type",          "Type  (e.g. monarchy, republic)",    "entry", False),
        ("_attr_leader",        "Current Leader",                     "entry", False),
        ("_attr_territory",     "Territory",                          "entry", False),
        ("_attr_policies",      "Key Policies  (one per line)",       "list",  False),
        ("_attr_alliances",     "Alliances  (one per line)",          "list",  False),
    ],
    "Economy": [
        ("name",        "Name  (trade system, currency, etc.)",       "entry", True),
        ("description", "Description",                                "text",  True),
        ("_attr_currency",      "Currency",                           "entry", False),
        ("_attr_main_exports",  "Main Exports  (one per line)",       "list",  False),
        ("_attr_main_imports",  "Main Imports  (one per line)",       "list",  False),
        ("_attr_trade_routes",  "Trade Routes  (one per line)",       "list",  False),
        ("_attr_guilds",        "Major Guilds / Factions  (one per line)", "list", False),
    ],
    "Technology": [
        ("name",        "Name",                                       "entry", True),
        ("description", "Description",                                "text",  True),
        ("_attr_era",           "Era / Time Period",                  "entry", False),
        ("_attr_capabilities",  "Capabilities  (one per line)",       "list",  False),
        ("_attr_limitations",   "Limitations  (one per line)",        "list",  False),
        ("_attr_creators",      "Creators / Maintainers  (one per line)", "list", False),
    ],
    "Item": [
        ("name",        "Name",                                       "entry", True),
        ("description", "Description",                                "text",  True),
        ("_attr_type",          "Type  (weapon, artifact, etc.)",     "entry", False),
        ("_attr_powers",        "Powers / Properties  (one per line)","list",  False),
        ("_attr_weaknesses",    "Weaknesses  (one per line)",         "list",  False),
        ("_attr_current_owner", "Current Owner / Location",           "entry", False),
        ("_attr_history",       "History  (one per line)",            "list",  False),
    ],
    "Event": [
        ("name",        "Name",                                       "entry", True),
        ("description", "Description",                                "text",  True),
        ("_attr_date",          "Date / Time Period",                 "entry", False),
        ("_attr_location",      "Location",                          "entry", False),
        ("_attr_participants",  "Participants  (one per line)",       "list",  False),
        ("_attr_consequences",  "Consequences  (one per line)",       "list",  False),
        ("_attr_related_plots", "Related Plotlines  (one per line)",  "list",  False),
    ],
}

# Entity types a canon rule can reference (and their display labels).
_TYPE_LABEL = {
    "character": "Character", "location": "Location", "plotline": "Plotline",
    "world_rule": "World Rule", "world_lore": "World Lore",
}

# Entity types that map to WorldEntity in the DB
_WORLD_ENTITY_TYPES = {
    "Organization", "Culture", "Species", "Skill",
    "Religion", "Politics", "Economy", "Technology", "Item", "Event",
}

ENTITY_TYPES = list(FIELD_SPECS.keys())


# ---------------------------------------------------------------------------
# Helpers to convert between widget values and Python values
# ---------------------------------------------------------------------------

def _read_list(widget: ctk.CTkTextbox) -> list[str]:
    raw = widget.get("1.0", "end").strip()
    return [line.strip() for line in raw.splitlines() if line.strip()]


def _read_dict(widget: ctk.CTkTextbox) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in widget.get("1.0", "end").splitlines():
        line = line.strip()
        if ":" in line:
            k, _, v = line.partition(":")
            out[k.strip()] = v.strip()
    return out


def _write_text(widget, value: Any) -> None:
    """Set a CTkTextbox value."""
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    if value:
        widget.insert("1.0", str(value))


def _write_list(widget: ctk.CTkTextbox, items: list) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", "\n".join(str(i) for i in items))


def _write_dict(widget: ctk.CTkTextbox, d: dict) -> None:
    widget.configure(state="normal")
    widget.delete("1.0", "end")
    widget.insert("1.0", "\n".join(f"{k}: {v}" for k, v in d.items()))


# ---------------------------------------------------------------------------
# Main tab class
# ---------------------------------------------------------------------------

class WorldBibleTab:
    def __init__(
        self,
        parent_tab,
        story_id: str,
        db_path: Path,
        chroma_path: Path,
    ):
        self.story_id  = story_id
        self.db_path   = db_path
        self.chroma_path = chroma_path

        self._current_entity = None   # Pydantic model being edited, or None for new
        self._field_widgets: dict[str, Any] = {}
        self._entity_buttons: list[ctk.CTkButton] = []

        self._build(parent_tab)
        self._on_type_change("Character")

    # ── Layout ────────────────────────────────────────────────────────────

    def _build(self, tab):
        tab.grid_rowconfigure(0, weight=1)
        tab.grid_columnconfigure(0, weight=0)
        tab.grid_columnconfigure(1, weight=1)

        # ── Left panel ────────────────────────────────────────────────
        left = ctk.CTkFrame(tab, width=190)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))
        left.grid_rowconfigure(2, weight=1)
        left.grid_propagate(False)

        self._type_var = ctk.StringVar(value="Character")
        self._type_menu = ctk.CTkOptionMenu(
            left,
            values=ENTITY_TYPES,
            variable=self._type_var,
            command=self._on_type_change,
            font=("", 12),
        )
        self._type_menu.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))

        add_btn = ctk.CTkButton(
            left, text="+ Add New", height=30,
            command=self._new_entity,
            fg_color=("gray70", "gray30"),
            hover_color=("gray60", "gray40"),
            font=("", 12),
        )
        add_btn.grid(row=1, column=0, sticky="ew", padx=8, pady=(0, 6))

        self._list_frame = ctk.CTkScrollableFrame(left, label_text="")
        self._list_frame.grid(row=2, column=0, sticky="nsew", padx=4, pady=(0, 8))

        # ── Right panel ───────────────────────────────────────────────
        right = ctk.CTkFrame(tab)
        right.grid(row=0, column=1, sticky="nsew")
        right.grid_rowconfigure(1, weight=1)
        right.grid_columnconfigure(0, weight=1)

        self._form_title = ctk.CTkLabel(
            right, text="", font=("", 15, "bold"), anchor="w",
        )
        self._form_title.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 4))

        self._form_frame = ctk.CTkScrollableFrame(right)
        self._form_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 4))
        self._form_frame.grid_columnconfigure(0, weight=1)

        # Save / Delete buttons
        btn_row = ctk.CTkFrame(right, fg_color="transparent")
        btn_row.grid(row=2, column=0, sticky="ew", padx=8, pady=(0, 8))

        self._save_btn = ctk.CTkButton(
            btn_row, text="Save", width=100, command=self._save,
        )
        self._save_btn.pack(side="left", padx=(0, 8))

        self._delete_btn = ctk.CTkButton(
            btn_row, text="Delete", width=100, command=self._delete,
            fg_color="#8B0000", hover_color="#a00000",
        )
        self._delete_btn.pack(side="left")

        self._error_label = ctk.CTkLabel(
            btn_row, text="", text_color="#f5a623", font=("", 11),
        )
        self._error_label.pack(side="left", padx=12)

    # ── Entity list ───────────────────────────────────────────────────────

    def _on_type_change(self, entity_type: str):
        self._current_entity = None
        self._rebuild_list(entity_type)
        self._rebuild_form(entity_type, entity=None)

    def _rebuild_list(self, entity_type: str):
        for btn in self._entity_buttons:
            btn.destroy()
        self._entity_buttons.clear()
        self._list_frame.configure(label_text=entity_type + "s")

        entities = self._load_all(entity_type)
        name_map = (
            {(t, i): n for t, i, n in self._all_ref_entities()}
            if entity_type == "Canon Rule" else None
        )
        for entity in entities:
            if entity_type == "Canon Rule":
                trig = name_map.get((entity.trigger_entity_type, entity.trigger_entity_id),
                                    entity.trigger_entity_id[:8])
                inj = name_map.get((entity.inject_entity_type, entity.inject_entity_id),
                                   entity.inject_entity_id[:8])
                label = f"{trig}  ▸  {inj}"
            else:
                label = getattr(entity, "name", None) or getattr(entity, "title", None) \
                        or ("Story Outline" if entity_type == "Story Outline" else f"{entity_type} (no name)")
            btn = ctk.CTkButton(
                self._list_frame,
                text=label,
                anchor="w",
                fg_color="transparent",
                hover_color=("gray75", "gray30"),
                font=("", 12),
                command=lambda e=entity: self._select_entity(e),
            )
            btn.pack(fill="x", pady=2)
            self._entity_buttons.append(btn)

    # ── Form ──────────────────────────────────────────────────────────────

    def _ref_candidates(self, ref_type: str) -> list[tuple[str, str]]:
        """(id, name) pairs for a referenced entity type, sorted by name, so
        linking fields offer a by-name picker instead of raw UUID entry."""
        import db
        try:
            if ref_type == "character":
                items = [(c.id, c.name) for c in db.get_all_characters(self.story_id, self.db_path)]
            elif ref_type == "location":
                items = [(l.id, l.name) for l in db.get_all_locations(self.story_id, self.db_path)]
            elif ref_type == "plotline":
                items = [(p.id, p.name) for p in db.get_all_plotlines(self.story_id, self.db_path)]
            else:
                items = []
        except Exception:
            items = []
        return sorted(items, key=lambda t: (t[1] or "").lower())

    @staticmethod
    def _ref_label(name: str, entity_id: str) -> str:
        return f"{name or '(unnamed)'}  ·  {entity_id[:8]}"

    def _all_ref_entities(self) -> list[tuple[str, str, str]]:
        """(type, id, name) across every entity type a canon rule can reference."""
        import db
        out: list[tuple[str, str, str]] = []
        try:
            out += [("character", c.id, c.name) for c in db.get_all_characters(self.story_id, self.db_path)]
            out += [("location", l.id, l.name) for l in db.get_all_locations(self.story_id, self.db_path)]
            out += [("plotline", p.id, p.name) for p in db.get_all_plotlines(self.story_id, self.db_path)]
            out += [("world_rule", r.id, r.title) for r in db.get_all_world_rules(self.db_path)]
            out += [("world_lore", w.id, w.title) for w in db.get_all_world_lore(self.db_path)]
        except Exception:
            pass
        return out

    def _rebuild_form(self, entity_type: str, entity=None):
        for widget in self._form_frame.winfo_children():
            widget.destroy()
        self._field_widgets.clear()
        self._error_label.configure(text="")

        is_new = entity is None
        self._form_title.configure(
            text=f"{'New ' if is_new else 'Edit '}{entity_type}"
        )
        self._delete_btn.configure(
            state="disabled" if is_new else "normal"
        )

        specs = FIELD_SPECS[entity_type]
        row = 0
        for spec in specs:
            field_name, label, wtype = spec[0], spec[1], spec[2]
            required = spec[3]
            choices = spec[4] if len(spec) > 4 else []

            req_mark = " *" if required else ""
            lbl = ctk.CTkLabel(
                self._form_frame,
                text=label + req_mark,
                anchor="w",
                font=("", 12),
            )
            lbl.grid(row=row, column=0, sticky="w", pady=(8, 0))
            row += 1

            # WorldEntity types store extra fields in .attributes dict
            if field_name.startswith("_attr_") and entity is not None:
                attr_key = field_name[6:]  # strip "_attr_"
                current_val = entity.attributes.get(attr_key)
            elif field_name.startswith("_attr_"):
                current_val = None
            elif wtype == "entity_any" and entity is not None:
                # Canon-rule references live as (type, id) pairs on the model.
                current_val = (getattr(entity, f"{field_name}_entity_type", ""),
                               getattr(entity, f"{field_name}_entity_id", ""))
            else:
                current_val = getattr(entity, field_name, None) if entity else None

            if wtype == "entry":
                w = ctk.CTkEntry(self._form_frame, font=("", 12))
                w.grid(row=row, column=0, sticky="ew", pady=(0, 2))
                if current_val is not None:
                    w.insert(0, str(current_val))

            elif wtype in ("text", "list", "dict"):
                h = 80 if wtype == "text" else 60
                w = ctk.CTkTextbox(self._form_frame, height=h, font=("", 12), wrap="word")
                w.grid(row=row, column=0, sticky="ew", pady=(0, 2))
                if current_val is not None:
                    if wtype == "list":
                        _write_list(w, current_val)
                    elif wtype == "dict":
                        _write_dict(w, current_val)
                    else:
                        _write_text(w, current_val)

            elif wtype == "option":
                var = ctk.StringVar(value=str(current_val) if current_val else choices[0])
                w = ctk.CTkOptionMenu(
                    self._form_frame, values=choices, variable=var, font=("", 12),
                )
                w.grid(row=row, column=0, sticky="ew", pady=(0, 2))
                w._var = var  # keep reference

            elif wtype == "bool":
                default_bool = current_val if current_val is not None else True
                var = ctk.StringVar(value=str(bool(default_bool)))
                w = ctk.CTkOptionMenu(
                    self._form_frame, values=["True", "False"], variable=var, font=("", 12),
                )
                w.grid(row=row, column=0, sticky="ew", pady=(0, 2))
                w._var = var  # keep reference

            elif wtype == "int":
                w = ctk.CTkEntry(self._form_frame, font=("", 12))
                w.grid(row=row, column=0, sticky="ew", pady=(0, 2))
                if current_val is not None:
                    w.insert(0, str(current_val))

            elif wtype == "ref":
                # Single entity reference: pick by name, store the id. `choices`
                # holds the referenced entity type (e.g. "location").
                candidates = self._ref_candidates(choices)
                label_to_id = {"(none)": ""}
                values = ["(none)"]
                cur_id = str(current_val) if current_val else ""
                cur_label = "(none)"
                for cid, name in candidates:
                    lab = self._ref_label(name, cid)
                    label_to_id[lab] = cid
                    values.append(lab)
                    if cid == cur_id:
                        cur_label = lab
                # Keep a stale/unknown current id visible instead of silently dropping it
                if cur_id and cur_id not in label_to_id.values():
                    lab = self._ref_label("(unknown)", cur_id)
                    label_to_id[lab] = cur_id
                    values.append(lab)
                    cur_label = lab
                var = ctk.StringVar(value=cur_label)
                w = ctk.CTkOptionMenu(self._form_frame, values=values, variable=var, font=("", 12))
                w.grid(row=row, column=0, sticky="ew", pady=(0, 2))
                w._var = var
                w._label_to_id = label_to_id

            elif wtype == "ref_list":
                # Multi reference: a checkbox per candidate; stores a list of ids.
                candidates = self._ref_candidates(choices)
                cur_ids = {str(x) for x in (current_val or [])}
                box = ctk.CTkFrame(self._form_frame, fg_color=("gray90", "gray17"))
                box.grid(row=row, column=0, sticky="ew", pady=(0, 2))
                checks: list[tuple[str, ctk.BooleanVar]] = []
                known = {cid for cid, _ in candidates}
                if not candidates and not cur_ids:
                    ctk.CTkLabel(box, text="(no characters to link yet)",
                                 text_color="gray60", font=("", 11)).pack(anchor="w", padx=6, pady=4)
                for cid, name in candidates:
                    bvar = ctk.BooleanVar(value=cid in cur_ids)
                    ctk.CTkCheckBox(box, text=self._ref_label(name, cid), variable=bvar,
                                    font=("", 11), checkbox_width=16, checkbox_height=16).pack(anchor="w", padx=6, pady=1)
                    checks.append((cid, bvar))
                for sid in cur_ids - known:  # stale ids kept and pre-checked
                    bvar = ctk.BooleanVar(value=True)
                    ctk.CTkCheckBox(box, text=self._ref_label("(unknown)", sid), variable=bvar,
                                    font=("", 11), checkbox_width=16, checkbox_height=16).pack(anchor="w", padx=6, pady=1)
                    checks.append((sid, bvar))
                w = box
                w._checks = checks

            elif wtype == "entity_any":
                # Pick any entity across all canon-referenceable types; stores (type, id).
                label_to_ref = {"(none)": None}
                values = ["(none)"]
                cur = current_val if (current_val and current_val[1]) else None
                cur_label = "(none)"
                for etype, eid, name in self._all_ref_entities():
                    lab = f"[{_TYPE_LABEL.get(etype, etype)}] {name or '(unnamed)'}  ·  {eid[:8]}"
                    label_to_ref[lab] = (etype, eid)
                    values.append(lab)
                    if cur and etype == cur[0] and eid == cur[1]:
                        cur_label = lab
                if cur and cur_label == "(none)":  # stale reference — keep it visible
                    lab = f"[{_TYPE_LABEL.get(cur[0], cur[0])}] (unknown)  ·  {cur[1][:8]}"
                    label_to_ref[lab] = cur
                    values.append(lab)
                    cur_label = lab
                var = ctk.StringVar(value=cur_label)
                w = ctk.CTkOptionMenu(self._form_frame, values=values, variable=var, font=("", 12))
                w.grid(row=row, column=0, sticky="ew", pady=(0, 2))
                w._var = var
                w._label_to_ref = label_to_ref

            self._field_widgets[field_name] = (wtype, w, required, choices)
            row += 1

    def _select_entity(self, entity):
        self._current_entity = entity
        entity_type = self._type_var.get()
        self._rebuild_form(entity_type, entity=entity)

    def _new_entity(self):
        self._current_entity = None
        entity_type = self._type_var.get()
        self._rebuild_form(entity_type, entity=None)

    # ── Validation ────────────────────────────────────────────────────────

    def _collect_and_validate(self) -> tuple[dict | None, str]:
        """Returns (data_dict, error_message). error_message is "" on success."""
        data: dict = {}
        for field_name, (wtype, widget, required, choices) in self._field_widgets.items():
            if wtype == "entry":
                val = widget.get().strip()
            elif wtype == "int":
                raw = widget.get().strip()
                if raw:
                    try:
                        val = int(raw)
                    except ValueError:
                        return None, f"'{field_name}' must be a whole number."
                else:
                    val = None
            elif wtype == "option":
                val = widget._var.get()
            elif wtype == "bool":
                val = widget._var.get() == "True"
            elif wtype == "list":
                val = _read_list(widget)
            elif wtype == "dict":
                val = _read_dict(widget)
            elif wtype == "text":
                val = widget.get("1.0", "end").strip()
            elif wtype == "ref":
                val = widget._label_to_id.get(widget._var.get(), "") or None
            elif wtype == "ref_list":
                val = [cid for cid, bvar in widget._checks if bvar.get()]
            elif wtype == "entity_any":
                val = widget._label_to_ref.get(widget._var.get())  # (type, id) or None
            else:
                val = None

            if required and (val is None or val == "" or val == [] or val == {}):
                return None, f"'{field_name}' is required."
            data[field_name] = val

        return data, ""

    # ── Save ──────────────────────────────────────────────────────────────

    def _save(self):
        self._error_label.configure(text="")
        data, err = self._collect_and_validate()
        if err:
            self._error_label.configure(text=err)
            return

        entity_type = self._type_var.get()
        try:
            self._write_entity(entity_type, data)
            self._rebuild_list(entity_type)
            self._error_label.configure(text="Saved.", text_color="#4caf50")
        except Exception as e:
            self._error_label.configure(text=f"Save failed: {e}", text_color="#f5a623")

    def _write_kwargs(self, data: dict) -> dict:
        """Upsert kwargs: drop blank scalar/list fields (so the form never wipes
        a field it didn't touch), but KEEP reference fields even when empty —
        that's what lets choosing '(none)' or unchecking all actually unlink."""
        ref_fields = {f for f, (wt, *_rest) in self._field_widgets.items() if wt in ("ref", "ref_list")}
        out: dict = {}
        for k, v in data.items():
            if k in ref_fields or v not in (None, "", [], {}):
                out[k] = v
        return out

    def _write_entity(self, entity_type: str, data: dict):
        import db
        import vector_store as vs
        from embeddings import get_default_embedder

        chroma = vs.get_chroma_client(self.chroma_path)
        emb = get_default_embedder()

        def embed(etype, eid, text, sid=None):
            vec = emb.embed(text)
            vs.upsert_entity(chroma, etype, eid, vec, text, story_id=sid)

        if entity_type == "Character":
            from schema import Character
            existing = self._current_entity
            kwargs = self._write_kwargs(data)
            if existing:
                obj = existing.model_copy(update=kwargs)
            else:
                obj = Character(**kwargs)
            db.upsert_character(obj, self.story_id, self.db_path)
            embed("character", obj.id, vs.character_text(obj), sid=self.story_id)

        elif entity_type == "Location":
            from schema import Location
            existing = self._current_entity
            kwargs = self._write_kwargs(data)
            if existing:
                obj = existing.model_copy(update=kwargs)
            else:
                obj = Location(**kwargs)
            db.upsert_location(obj, self.story_id, self.db_path)
            embed("location", obj.id, vs.location_text(obj), sid=self.story_id)

        elif entity_type == "Plotline":
            from schema import Plotline
            existing = self._current_entity
            tension = data.get("current_tension")
            if tension is not None:
                data["current_tension"] = max(0, min(10, int(tension)))
            kwargs = self._write_kwargs(data)
            if existing:
                obj = existing.model_copy(update=kwargs)
            else:
                obj = Plotline(**kwargs)
            db.upsert_plotline(obj, self.story_id, self.db_path)
            embed("plotline", obj.id, vs.plotline_text(obj), sid=self.story_id)

        elif entity_type == "World Rule":
            from schema import WorldRule
            existing = self._current_entity
            kwargs = self._write_kwargs(data)
            if existing:
                obj = existing.model_copy(update=kwargs)
            else:
                obj = WorldRule(**kwargs)
            db.upsert_world_rule(obj, self.db_path)
            embed("world_rule", obj.id, vs.world_rule_text(obj))

        elif entity_type == "World Lore":
            from schema import WorldLore
            existing = self._current_entity
            kwargs = self._write_kwargs(data)
            if existing:
                obj = existing.model_copy(update=kwargs)
            else:
                obj = WorldLore(**kwargs)
            db.upsert_world_lore(obj, self.db_path)
            embed("world_lore", obj.id, vs.world_lore_text(obj))

        elif entity_type == "POV State":
            from schema import POVState
            existing = self._current_entity
            kwargs = self._write_kwargs(data)
            if existing:
                obj = existing.model_copy(update=kwargs)
            else:
                obj = POVState(**kwargs)
            db.upsert_pov_state(obj, self.story_id, self.db_path)

        elif entity_type == "Story Outline":
            from datetime import datetime
            from schema import StoryOutline
            # Empty fields are applied (clearing the premise is a valid edit),
            # so no blank-value filtering here. Beats/arcs are left untouched.
            updates = {
                "premise":        data.get("premise") or "",
                "theme":          data.get("theme") or "",
                "planned_ending": data.get("planned_ending") or "",
                "updated_at":     datetime.utcnow(),
            }
            existing = self._current_entity or db.get_story_outline(self.story_id, self.db_path)
            if existing:
                obj = existing.model_copy(update=updates)
            else:
                obj = StoryOutline(story_id=self.story_id, **updates)
            db.upsert_story_outline(obj, self.db_path)

        elif entity_type == "Canon Rule":
            import uuid
            from schema import CanonRule
            trigger, inject = data.get("trigger"), data.get("inject")
            if not trigger or not inject:
                raise ValueError("pick both a trigger and an inject entity")
            tt, tid = trigger
            it, iid = inject
            existing = self._current_entity
            rule_id = existing.rule_id if existing else f"rule-{uuid.uuid4().hex[:12]}"
            obj = CanonRule(
                rule_id=rule_id, story_id=self.story_id,
                trigger_entity_type=tt, trigger_entity_id=tid,
                inject_entity_type=it, inject_entity_id=iid,
                reason=data.get("reason") or "",
            )
            db.insert_canon_rule(obj, self.db_path)

        elif entity_type in _WORLD_ENTITY_TYPES:
            from schema import WorldEntity
            existing = self._current_entity
            # Split fields: regular (name, description) vs _attr_* → attributes dict
            base: dict = {}
            attrs: dict = {}
            for k, v in data.items():
                if k.startswith("_attr_"):
                    attr_key = k[6:]
                    if v not in (None, "", [], {}):
                        attrs[attr_key] = v
                elif v not in (None, "", [], {}):
                    base[k] = v
            category = entity_type  # use the entity type name as the category string
            if existing:
                merged_attrs = {**existing.attributes, **attrs}
                obj = existing.model_copy(update={**base, "attributes": merged_attrs})
            else:
                obj = WorldEntity(category=category, attributes=attrs, **base)
            db.upsert_world_entity(obj, self.story_id, self.db_path)
            embed("world_entity", obj.id, vs.world_entity_text(obj), sid=self.story_id)

        # Keep the saved object as the current entity so a second Save press
        # updates it in place (same id) instead of inserting a duplicate.
        self._current_entity = obj

    # ── Delete ────────────────────────────────────────────────────────────

    def _delete(self):
        if self._current_entity is None:
            return
        entity_type = self._type_var.get()

        dlg = ctk.CTkInputDialog(
            text=f"Type DELETE to confirm removing this {entity_type}.",
            title="Confirm Delete",
        )
        confirm = dlg.get_input()
        if confirm != "DELETE":
            return

        try:
            self._delete_entity(entity_type, self._current_entity)
            self._current_entity = None
            self._rebuild_list(entity_type)
            self._rebuild_form(entity_type, entity=None)
            self._error_label.configure(text="Deleted.", text_color="#4caf50")
        except Exception as e:
            self._error_label.configure(text=f"Delete failed: {e}", text_color="#f5a623")

    def _delete_entity(self, entity_type: str, entity):
        import db
        conn = db.get_connection(self.db_path)
        table_map = {
            "Character": ("characters", True),
            "Location":  ("locations",  True),
            "Plotline":  ("plotlines",  True),
            "World Rule":("world_rules",False),
            "World Lore":("world_lore", False),
        }
        if entity_type == "POV State":
            conn.execute("DELETE FROM pov_state WHERE story_id = ?", (self.story_id,))
        elif entity_type == "Story Outline":
            # Removing the outline makes the next generation re-create it from
            # scratch (LLM init on cold start, or a fresh empty outline mid-story).
            conn.execute("DELETE FROM story_outline WHERE story_id = ?", (self.story_id,))
        elif entity_type in table_map:
            table, scoped = table_map[entity_type]
            eid = getattr(entity, "id", None) or getattr(entity, "rule_id", None)
            if scoped:
                conn.execute(f"DELETE FROM {table} WHERE id = ? AND story_id = ?",
                             (eid, self.story_id))
            else:
                conn.execute(f"DELETE FROM {table} WHERE id = ?", (eid,))
        elif entity_type == "Canon Rule":
            conn.execute("DELETE FROM canon_rules WHERE rule_id = ? AND story_id = ?",
                         (entity.rule_id, self.story_id))
        elif entity_type in _WORLD_ENTITY_TYPES:
            conn.execute("DELETE FROM world_entities WHERE id = ? AND story_id = ?",
                         (entity.id, self.story_id))
        conn.commit()
        conn.close()

    # ── DB loaders ────────────────────────────────────────────────────────

    def _load_all(self, entity_type: str) -> list:
        import db
        try:
            db.init_db(self.db_path)
            if entity_type == "Character":
                return db.get_all_characters(self.story_id, self.db_path)
            elif entity_type == "Location":
                return db.get_all_locations(self.story_id, self.db_path)
            elif entity_type == "Plotline":
                return db.get_all_plotlines(self.story_id, self.db_path)
            elif entity_type == "World Rule":
                return db.get_all_world_rules(self.db_path)
            elif entity_type == "World Lore":
                return db.get_all_world_lore(self.db_path)
            elif entity_type == "POV State":
                pov = db.get_pov_state(self.story_id, self.db_path)
                return [pov] if pov else []
            elif entity_type == "Story Outline":
                outline = db.get_story_outline(self.story_id, self.db_path)
                return [outline] if outline else []
            elif entity_type == "Canon Rule":
                return db.get_all_canon_rules(self.story_id, self.db_path)
            elif entity_type in _WORLD_ENTITY_TYPES:
                return db.get_world_entities_by_category(entity_type, self.story_id, self.db_path)
        except Exception:
            return []
        return []
