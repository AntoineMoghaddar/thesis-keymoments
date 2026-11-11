"""Window of the telemetry logger. Start with: python -m logger.app"""
from __future__ import annotations

import json
import re
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from km.config import load_config

from . import questionnaire as qn
from .acc_shm import STATUS_LIVE, STATUS_PAUSE, STATUS_REPLAY, AccSource
from .i18n import TEXT, tr
from .mock import MockSource
from .recorder import Recorder
from .transfer import transfer

SETTINGS_FILE = Path.home() / ".km_logger.json"
PSEUDONYM = re.compile(r"^[A-Z]\d{1,3}$")
LEVELS = ["none", "moderate", "high"]


def format_lap(ms: int) -> str:
    if ms <= 0 or ms > 3_600_000:
        return "-"
    minutes, rest = divmod(ms / 1000.0, 60)
    return f"{int(minutes)}:{rest:06.3f}"


class QuestionnaireWindow(tk.Toplevel):
    """Scrollable form with the experience questionnaire."""

    def __init__(self, app: "LoggerApp", driver: str):
        super().__init__(app.root)
        self.app, self.driver, self.lang = app, driver, app.lang.get()
        self.title(f"{tr('questionnaire', self.lang)} {driver}")
        self.geometry("760x720")
        self.vars: dict[str, object] = {}
        previous = qn.load_participant(app.data_dir.get(), driver).get("questionnaire", {})

        canvas = tk.Canvas(self, highlightthickness=0)
        bar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        body = ttk.Frame(canvas, padding=16)
        body.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=body, anchor="nw")
        canvas.configure(yscrollcommand=bar.set)
        canvas.pack(side="left", fill="both", expand=True)
        bar.pack(side="right", fill="y")
        canvas.bind_all("<MouseWheel>", lambda e: canvas.yview_scroll(int(-e.delta / 120), "units"))
        self.bind("<Destroy>", lambda e: canvas.unbind_all("<MouseWheel>") if e.widget is self else None)

        ttk.Label(body, text=tr("q_intro", self.lang), wraplength=680).pack(anchor="w", pady=(0, 8))
        for q in qn.QUESTIONS:
            if "section" in q:
                ttk.Label(body, text=q["section"][self.lang], font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(14, 4))
                continue
            ttk.Label(body, text=q[self.lang], wraplength=680).pack(anchor="w", pady=(8, 2))
            old = previous.get(q["id"])
            if q["kind"] == "choice":
                var = tk.StringVar()
                options = qn.option_text(q, self.lang)
                for en, shown in zip(qn.option_text(q, "en"), options):
                    if old == en:
                        var.set(shown)
                ttk.Combobox(body, textvariable=var, values=options, state="readonly", width=70).pack(anchor="w")
            elif q["kind"] == "multi":
                var = {}
                for en, shown in zip(qn.option_text(q, "en"), qn.option_text(q, self.lang)):
                    var[en] = tk.BooleanVar(value=bool(old and en in old))
                    ttk.Checkbutton(body, text=shown, variable=var[en]).pack(anchor="w", padx=12)
            elif q["kind"] == "scale":
                var = tk.IntVar(value=int(old) if old else 0)
                row = ttk.Frame(body)
                row.pack(anchor="w")
                for value in range(1, 6):
                    ttk.Radiobutton(row, text=str(value), value=value, variable=var).pack(side="left", padx=8)
            else:
                var = tk.StringVar(value=old or "")
                ttk.Entry(body, textvariable=var, width=73).pack(anchor="w")
            self.vars[q["id"]] = var

        buttons = ttk.Frame(body)
        buttons.pack(anchor="e", pady=18)
        ttk.Button(buttons, text=tr("cancel", self.lang), command=self.destroy).pack(side="left", padx=6)
        ttk.Button(buttons, text=tr("save", self.lang), command=self.save).pack(side="left")
        self.transient(app.root)
        self.grab_set()

    def answers(self) -> dict:
        out = {}
        for q in qn.ITEMS:
            var = self.vars[q["id"]]
            if q["kind"] == "choice":
                out[q["id"]] = qn.to_english(q, var.get()) or None
            elif q["kind"] == "multi":
                out[q["id"]] = [en for en, flag in var.items() if flag.get()]
            elif q["kind"] == "scale":
                out[q["id"]] = var.get() or None
            else:
                out[q["id"]] = var.get().strip()
        return out

    def save(self) -> None:
        answers = self.answers()
        open_items = qn.missing(answers)
        if open_items:
            messagebox.showwarning(tr("questionnaire", self.lang), tr("q_missing", self.lang) + ", ".join(open_items),
                                   parent=self)
            return
        qn.save_participant(self.app.data_dir.get(), self.driver, questionnaire=answers,
                            questionnaire_language=self.lang)
        self.destroy()
        self.app.refresh_participant()


class LoggerApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.cfg = load_config()
        saved = json.loads(SETTINGS_FILE.read_text(encoding="utf-8")) if SETTINGS_FILE.exists() else {}
        self.lang = tk.StringVar(value=saved.get("lang", "nl"))
        self.data_dir = tk.StringVar(value=saved.get("data_dir", str(Path.home() / "km_data")))
        self.target_dir = tk.StringVar(value=saved.get("target_dir", ""))
        self.driver = tk.StringVar()
        self.level = tk.StringVar()
        self.consent = tk.BooleanVar(value=False)
        self.publication = tk.BooleanVar(value=False)
        self.circuit = tk.StringVar(value=self.cfg["study"]["circuits"][0])
        self.number = tk.IntVar(value=1)
        self.demo = tk.BooleanVar(value=False)
        self.delete_local = tk.BooleanVar(value=False)
        self.recorder: Recorder | None = None
        self.rec_source = None
        self.message = tk.StringVar()
        self.status_vars = {k: tk.StringVar(value="-") for k in
                            ("sim", "car", "track", "recording", "laps", "last_lap", "samples", "rate", "elapsed", "lap_state", "q")}
        try:
            self.acc = AccSource()
        except OSError:
            self.acc = None
        self.driver.trace_add("write", lambda *_: self.refresh_participant())
        self.build()
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.tick()

    # -- layout -------------------------------------------------------------
    def build(self) -> None:
        for child in self.root.winfo_children():
            child.destroy()
        lang = self.lang.get()
        self.root.title(tr("title", lang))
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill="both", expand=True)

        top = ttk.Frame(main)
        top.pack(fill="x")
        ttk.Label(top, text=tr("title", lang), font=("Segoe UI", 15, "bold")).pack(side="left")
        for code in ("en", "nl"):
            ttk.Radiobutton(top, text=code.upper(), value=code, variable=self.lang, command=self.build).pack(side="right")
        ttk.Label(top, text=tr("language", lang) + ":").pack(side="right", padx=6)

        box = ttk.LabelFrame(main, text=tr("participant", lang), padding=10)
        box.pack(fill="x", pady=8)
        ttk.Label(box, text=tr("pseudonym", lang)).grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.driver, width=10).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(box, text=tr("level", lang)).grid(row=1, column=0, sticky="w", pady=4)
        names = TEXT_LEVELS(lang)
        self.level_box = ttk.Combobox(box, values=[names[k] for k in LEVELS], state="readonly", width=14)
        self.level_box.grid(row=1, column=1, sticky="w", padx=8)
        if self.level.get() in LEVELS:
            self.level_box.set(names[self.level.get()])
        self.level_box.bind("<<ComboboxSelected>>", lambda e: self.set_level())
        ttk.Checkbutton(box, text=tr("consent", lang), variable=self.consent, command=self.save_participant).grid(
            row=2, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(box, text=tr("publication", lang), variable=self.publication, command=self.save_participant).grid(
            row=3, column=0, columnspan=2, sticky="w")
        ttk.Button(box, text=tr("q_open", lang), command=self.open_questionnaire).grid(row=4, column=0, sticky="w", pady=(6, 0))
        ttk.Label(box, textvariable=self.status_vars["q"]).grid(row=4, column=1, sticky="w", padx=8, pady=(6, 0))

        box = ttk.LabelFrame(main, text=tr("session", lang), padding=10)
        box.pack(fill="x", pady=8)
        ttk.Label(box, text=tr("circuit", lang)).grid(row=0, column=0, sticky="w")
        ttk.Combobox(box, textvariable=self.circuit, values=self.cfg["study"]["circuits"], state="readonly",
                     width=18).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(box, text=tr("number", lang)).grid(row=0, column=2, sticky="w", padx=(16, 0))
        ttk.Spinbox(box, from_=1, to=9, textvariable=self.number, width=4).grid(row=0, column=3, sticky="w", padx=8)
        ttk.Label(box, text=tr("data_dir", lang)).grid(row=1, column=0, sticky="w", pady=4)
        ttk.Entry(box, textvariable=self.data_dir, width=48).grid(row=1, column=1, columnspan=3, sticky="w", padx=8)
        ttk.Button(box, text=tr("browse", lang), command=lambda: self.browse(self.data_dir)).grid(row=1, column=4)
        ttk.Checkbutton(box, text=tr("demo", lang), variable=self.demo).grid(row=2, column=0, columnspan=4, sticky="w")
        self.start_button = ttk.Button(box, text=tr("start", lang), command=self.start)
        self.start_button.grid(row=3, column=0, sticky="w", pady=(8, 0))
        self.stop_button = ttk.Button(box, text=tr("stop", lang), command=self.stop)
        self.stop_button.grid(row=3, column=1, sticky="w", pady=(8, 0), padx=8)

        box = ttk.LabelFrame(main, text=tr("status", lang), padding=10)
        box.pack(fill="x", pady=8)
        rows = ["sim", "car", "track", "recording", "laps", "last_lap", "lap_state", "elapsed", "samples", "rate"]
        for k, key in enumerate(rows):
            ttk.Label(box, text=tr(key, lang) + ":").grid(row=k // 2, column=(k % 2) * 2, sticky="w", padx=(0, 6), pady=1)
            ttk.Label(box, textvariable=self.status_vars[key], width=30).grid(row=k // 2, column=(k % 2) * 2 + 1, sticky="w")

        box = ttk.LabelFrame(main, text=tr("transfer", lang), padding=10)
        box.pack(fill="x", pady=8)
        ttk.Label(box, text=tr("target_dir", lang)).grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.target_dir, width=48).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Button(box, text=tr("browse", lang), command=lambda: self.browse(self.target_dir)).grid(row=0, column=2)
        ttk.Checkbutton(box, text=tr("delete_local", lang), variable=self.delete_local).grid(row=1, column=0, columnspan=2, sticky="w")
        ttk.Button(box, text=tr("do_transfer", lang), command=self.transfer).grid(row=2, column=0, columnspan=2, sticky="w", pady=(6, 0))

        ttk.Label(main, textvariable=self.message, wraplength=640, foreground="#1f4e79").pack(anchor="w", pady=(4, 0))
        self.refresh_participant()
        self.update_buttons()

    def update_buttons(self) -> None:
        running = self.recorder is not None
        self.start_button.configure(state="disabled" if running else "normal")
        self.stop_button.configure(state="normal" if running else "disabled")

    def browse(self, var: tk.StringVar) -> None:
        folder = filedialog.askdirectory(initialdir=var.get() or str(Path.home()))
        if folder:
            var.set(folder)
            self.save_settings()

    def save_settings(self) -> None:
        SETTINGS_FILE.write_text(json.dumps({"lang": self.lang.get(), "data_dir": self.data_dir.get(),
                                             "target_dir": self.target_dir.get()}), encoding="utf-8")

    # -- participant --------------------------------------------------------
    def pseudonym(self) -> str | None:
        value = self.driver.get().strip().upper()
        return value if PSEUDONYM.match(value) else None

    def refresh_participant(self) -> None:
        """Show what is already stored for the pseudonym that was typed."""
        lang, driver = self.lang.get(), self.pseudonym()
        record = qn.load_participant(self.data_dir.get(), driver) if driver else {}
        self.status_vars["q"].set(tr("q_done" if record.get("questionnaire") else "q_todo", lang))
        if driver and record.get("experience_level") in LEVELS:
            self.level.set(record["experience_level"])
            self.level_box.set(TEXT_LEVELS(lang)[self.level.get()])
        if driver and "consent_signed" in record:
            self.consent.set(bool(record["consent_signed"]))
        if driver:
            self.publication.set(bool(record.get("publication_consent", False)))
        if driver:
            folder = Path(self.data_dir.get()) / "raw" / driver / self.circuit.get()
            self.number.set(len(list(folder.glob("*.parquet"))) + 1 if folder.exists() else 1)

    def set_level(self) -> None:
        names = TEXT_LEVELS(self.lang.get())
        self.level.set(next(k for k in LEVELS if names[k] == self.level_box.get()))
        self.save_participant()

    def save_participant(self) -> None:
        driver = self.pseudonym()
        if driver:
            fields = {"consent_signed": bool(self.consent.get()), "publication_consent": bool(self.publication.get())}
            if self.level.get() in LEVELS:
                fields["experience_level"] = self.level.get()
            qn.save_participant(self.data_dir.get(), driver, **fields)

    def open_questionnaire(self) -> None:
        driver = self.pseudonym()
        if not driver:
            messagebox.showerror(tr("participant", self.lang.get()), tr("err_pseudonym", self.lang.get()))
            return
        QuestionnaireWindow(self, driver)

    # -- recording ----------------------------------------------------------
    def start(self) -> None:
        lang, driver = self.lang.get(), self.pseudonym()
        title = tr("session", lang)
        if not driver:
            return messagebox.showerror(title, tr("err_pseudonym", lang))
        if not self.consent.get():
            return messagebox.showerror(title, tr("err_consent", lang))
        if self.level.get() not in LEVELS:
            return messagebox.showerror(title, tr("err_level", lang))
        record = qn.load_participant(self.data_dir.get(), driver)
        if not record.get("questionnaire") and not messagebox.askyesno(title, tr("err_questionnaire", lang)):
            return
        if self.demo.get():
            source, circuit, expected_car = MockSource(), "synth_a", None
        else:
            source, circuit, expected_car = self.acc, self.circuit.get(), self.cfg["study"]["car_model"]
            if source is None or source.status() != STATUS_LIVE:
                return messagebox.showerror(title, tr("err_not_live", lang))
            info = source.info()
            if info["car_model"].lower() != expected_car.lower() and not messagebox.askyesno(
                    title, tr("warn_car", lang, found=info["car_model"], expected=expected_car)):
                return
            if circuit.lower() not in info["track"].lower() and not messagebox.askyesno(
                    title, tr("warn_track", lang, found=info["track"], expected=circuit)):
                return
        self.save_participant()
        self.save_settings()
        extra = dict(experience_level=self.level.get(), demo=bool(self.demo.get()), **source.info(), **source.settings())
        log = self.cfg["logger"]
        self.recorder = Recorder(source, self.data_dir.get(), driver, circuit, int(self.number.get()), hz=log["hz"],
                                 flush_seconds=log["flush_seconds"], expected_car=expected_car, extra_meta=extra)
        self.rec_source = source
        self.recorder.start()
        self.message.set(str(self.recorder.data_path))
        self.update_buttons()

    def stop(self) -> None:
        if self.recorder is None:
            return
        lang = self.lang.get()
        meta = self.recorder.stop()
        error = self.recorder.error
        self.recorder = None
        text = tr("saved", lang, laps=meta["laps_completed"], samples=meta["n_samples"], rate=meta["mean_rate_hz"])
        self.message.set(f"{tr('rec_error', lang)}{error} {text}" if error else text)
        self.update_buttons()
        self.refresh_participant()

    def transfer(self) -> None:
        lang = self.lang.get()
        if self.delete_local.get() and not messagebox.askyesno(tr("transfer", lang), tr("confirm_delete", lang)):
            return
        try:
            result = transfer(self.data_dir.get(), self.target_dir.get(), delete_source=self.delete_local.get())
        except (OSError, ValueError) as exc:
            return messagebox.showerror(tr("transfer", lang), str(exc))
        self.save_settings()
        self.message.set(tr("transfer_done", lang, n_failed=len(result["failed"]), **result))

    # -- status -------------------------------------------------------------
    def tick(self) -> None:
        lang, sv = self.lang.get(), self.status_vars
        try:
            if self.acc is None:
                sv["sim"].set(tr("sim_unavailable", lang))
            else:
                status = self.acc.status()
                names = {STATUS_LIVE: "sim_live", STATUS_PAUSE: "sim_pause", STATUS_REPLAY: "sim_replay"}
                sv["sim"].set(tr(names.get(status, "sim_off"), lang))
                info = self.acc.info()
                sv["car"].set(info["car_model"] or "-")
                sv["track"].set(info["track"] or "-")
        except Exception as exc:  # the status display must never stop a recording
            sv["sim"].set(str(exc))
        rec = self.recorder
        sv["recording"].set(tr("rec_on" if rec else "rec_off", lang))
        if rec:
            s = rec.stats
            target = self.cfg["study"]["laps_per_session"]
            sv["laps"].set(f"{s['laps_completed']}  ({tr('target', lang)} {target})")
            sv["last_lap"].set(format_lap(s["last_lap_ms"]))
            sv["samples"].set(f"{s['samples']:,}")
            sv["rate"].set(f"{s['rate']:.1f} Hz")
            sv["elapsed"].set(f"{int(s['elapsed'] // 60)}:{int(s['elapsed'] % 60):02d}")
            state = tr("in_pit", lang) if s["in_pit"] else tr("valid" if s["valid_lap"] else "invalid", lang)
            sv["lap_state"].set(state)
            if rec.error:
                self.stop()
        self.root.after(300, self.tick)

    def close(self) -> None:
        if self.recorder is not None:
            if not messagebox.askyesno(tr("title", self.lang.get()), tr("close_running", self.lang.get())):
                return
            self.stop()
        self.save_settings()
        self.root.destroy()


def TEXT_LEVELS(lang: str) -> dict:
    return TEXT["levels"][lang]


def main() -> None:
    root = tk.Tk()
    root.minsize(700, 640)
    LoggerApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
