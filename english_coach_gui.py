#!/usr/bin/env python3
"""
English Speaking Coach — desktop GUI
====================================

A simple native window (Tkinter, no extra installs) to:
  1. pick an audio/video recording
  2. paste or load the transcript / script you read aloud
  3. choose which analyses to run (Whisper+Claude, Azure pronunciation)
  4. generate the HTML report and open it in your browser

Run:
    python english_coach_gui.py

Keys can be typed into the form, or set beforehand as environment variables
ANTHROPIC_API_KEY, AZURE_SPEECH_KEY, AZURE_SPEECH_REGION.
"""

import json
import os
import queue
import shutil
import threading
import traceback
import webbrowser
from datetime import date

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import english_coach as ec


AUDIO_TYPES = [("Audio/Video", "*.m4a *.mp3 *.wav *.mp4 *.mov *.aac *.flac *.ogg"),
               ("All files", "*.*")]

CONFIG_PATH = os.path.join(os.path.expanduser("~"), ".english_coach.json")


def load_config():
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(cfg):
    try:
        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f)
    except Exception:
        pass


class CoachGUI:
    def __init__(self, root):
        self.root = root
        self.q = queue.Queue()
        self.busy = False
        root.title("English Speaking Coach")
        root.geometry("680x700")
        root.minsize(620, 600)

        pad = {"padx": 12, "pady": 6}
        main = ttk.Frame(root, padding=14)
        main.pack(fill="both", expand=True)
        main.columnconfigure(1, weight=1)
        r = 0

        ttk.Label(main, text="English Speaking Coach",
                  font=("Helvetica", 18, "bold")).grid(row=r, column=0, columnspan=3, sticky="w")
        r += 1
        ttk.Label(main, text="Upload a recording + the script you read, get a report.",
                  foreground="#666").grid(row=r, column=0, columnspan=3, sticky="w", pady=(0, 10))
        r += 1

        # --- Audio file ---
        ttk.Label(main, text="Audio file:").grid(row=r, column=0, sticky="w")
        self.audio_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.audio_var).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(main, text="Browse…", command=self.pick_audio).grid(row=r, column=2)
        r += 1

        # --- Transcript ---
        head = ttk.Frame(main)
        head.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(10, 0))
        ttk.Label(head, text="Transcript (the script you read aloud):").pack(side="left")
        ttk.Button(head, text="Load from file…", command=self.load_transcript).pack(side="right")
        ttk.Button(head, text="Transcribe audio ⮯",
                   command=self.transcribe_audio).pack(side="right", padx=(0, 6))
        r += 1
        self.transcript = tk.Text(main, height=7, wrap="word")
        self.transcript.grid(row=r, column=0, columnspan=3, sticky="ew")
        r += 1

        # --- Grammar analysis JSON (from Claude) ---
        ttk.Label(main, text="Grammar analysis JSON (from Claude, optional):").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(10, 0))
        r += 1
        self.json_var = tk.StringVar()
        ttk.Entry(main, textvariable=self.json_var).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(main, text="Browse…", command=self.pick_json).grid(row=r, column=2)
        ttk.Label(main, text="If set, the app uses this instead of Whisper+Claude (no API key needed).",
                  foreground="#888").grid(row=r, column=0, sticky="w")
        r += 1

        # --- Options ---
        opt = ttk.LabelFrame(main, text="What to run", padding=10)
        opt.grid(row=r, column=0, columnspan=3, sticky="ew", pady=12)
        opt.columnconfigure(1, weight=1)
        self.do_llm = tk.BooleanVar(value=True)
        self.do_azure = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt, text="Grammar, word choice & fluency  (Whisper + Claude)",
                        variable=self.do_llm).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Checkbutton(opt, text="Pronunciation scoring  (Azure — needs transcript above)",
                        variable=self.do_azure).grid(row=1, column=0, columnspan=2, sticky="w")

        # keys — env wins, then last-saved values, then defaults
        cfg = load_config()

        # scoring strictness (Azure runs generous; re-grade tougher)
        ttk.Label(opt, text="Scoring strictness:").grid(row=5, column=0, sticky="w", pady=(8, 0))
        self.strict_labels = {
            "Standard (Azure default)": "standard",
            "Strict": "strict",
            "Very strict": "very_strict",
        }
        self.strict_var = tk.StringVar(
            value=cfg.get("strictness_label", "Strict"))
        ttk.OptionMenu(opt, self.strict_var, self.strict_var.get(),
                       *self.strict_labels.keys()).grid(
            row=5, column=1, sticky="ew", pady=(8, 0))
        self.anthropic_key = tk.StringVar(
            value=os.environ.get("ANTHROPIC_API_KEY") or cfg.get("anthropic_key", ""))
        self.azure_key = tk.StringVar(
            value=os.environ.get("AZURE_SPEECH_KEY") or cfg.get("azure_key", ""))
        self.azure_region = tk.StringVar(
            value=os.environ.get("AZURE_SPEECH_REGION") or cfg.get("azure_region", "eastus"))
        ttk.Label(opt, text="Anthropic API key:").grid(row=2, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(opt, textvariable=self.anthropic_key, show="•").grid(row=2, column=1, sticky="ew", pady=(8, 0))
        ttk.Label(opt, text="Azure key:").grid(row=3, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.azure_key, show="•").grid(row=3, column=1, sticky="ew")
        ttk.Label(opt, text="Azure region:").grid(row=4, column=0, sticky="w")
        ttk.Entry(opt, textvariable=self.azure_region).grid(row=4, column=1, sticky="ew")
        r += 1

        # --- Output ---
        ttk.Label(main, text="Save report to:").grid(row=r, column=0, sticky="w")
        default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.html")
        self.out_var = tk.StringVar(value=default_out)
        ttk.Entry(main, textvariable=self.out_var).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(main, text="Browse…", command=self.pick_out).grid(row=r, column=2)
        r += 1

        # --- Run + status ---
        btns = ttk.Frame(main)
        btns.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(14, 6))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        self.run_btn = ttk.Button(btns, text="Analyze  ▶", command=self.run)
        self.run_btn.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self.regen_btn = ttk.Button(btns, text="Re-build dashboard  ♻",
                                    command=self.regenerate)
        self.regen_btn.grid(row=1, column=0, sticky="ew", padx=(0, 6))
        self.open_btn = ttk.Button(btns, text="Open dashboard  📄", command=self.open_report)
        self.open_btn.grid(row=1, column=1, sticky="ew")
        r += 1
        self.bar = ttk.Progressbar(main, mode="indeterminate")
        self.bar.grid(row=r, column=0, columnspan=3, sticky="ew")
        r += 1
        self.status = tk.StringVar(value="Ready.")
        ttk.Label(main, textvariable=self.status, foreground="#444").grid(
            row=r, column=0, columnspan=3, sticky="w", pady=(6, 0))

        self.root.after(150, self.poll)

    def set_audio(self, p, force=False):
        """Set the audio path and auto-load the matching .txt / .json — looking
        next to the file AND in this recording's library subfolder."""
        self.audio_var.set(p)
        base = os.path.splitext(os.path.basename(p))[0]
        # candidate locations for the transcript/json: same folder, or library/<stem>/
        dirs = [os.path.dirname(p), ec.rec_dir_for(base, create=False)]

        def find(ext):
            for d in dirs:
                cand = os.path.join(d, base + ext)
                if os.path.exists(cand):
                    return cand
            return None

        loaded = []
        txt = find(".txt")
        if txt and (force or not self.transcript.get("1.0", "end").strip()):
            try:
                with open(txt, encoding="utf-8") as f:
                    self.transcript.delete("1.0", "end")
                    self.transcript.insert("1.0", f.read())
                loaded.append(os.path.basename(txt))
            except Exception:
                pass
        js = find(".json")
        if js and (force or not self.json_var.get().strip()):
            self.json_var.set(js)
            loaded.append(os.path.basename(js))
        elif force:
            self.json_var.set("")
        self.out_var.set(os.path.splitext(p)[0] + ".report.html")
        self.status.set("Loaded: " + ", ".join(loaded) if loaded
                        else "Selected %s" % os.path.basename(p))

    # ---- file pickers ----
    def pick_audio(self):
        p = filedialog.askopenfilename(title="Choose a recording", filetypes=AUDIO_TYPES)
        if p:
            self.set_audio(p, force=True)

    def load_transcript(self):
        p = filedialog.askopenfilename(title="Choose transcript",
                                       filetypes=[("Text", "*.txt"), ("All files", "*.*")])
        if p:
            with open(p, encoding="utf-8") as f:
                self.transcript.delete("1.0", "end")
                self.transcript.insert("1.0", f.read())

    def dashboard_path(self):
        """The dashboard you open lives in the project root (next to this app)."""
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard.html")

    def transcribe_audio(self):
        """Run Whisper on the selected audio/video to auto-fill the transcript."""
        if self.busy:
            return
        audio = self.audio_var.get().strip()
        if not audio or not os.path.exists(audio):
            messagebox.showerror("Missing audio", "Choose an audio or video file first.")
            return
        self.busy = True
        self.run_btn.config(state="disabled")
        self.bar.start(12)
        self.status.set("Transcribing (Whisper)… first run downloads the model.")
        threading.Thread(target=self._transcribe_worker, args=(audio,),
                         daemon=True).start()

    def _transcribe_worker(self, audio):
        try:
            text, _words, _dur = ec.transcribe(
                audio, progress=lambda m: self.q.put(("status", m)))
            stem = os.path.splitext(audio)[0]
            try:
                with open(stem + ".txt", "w", encoding="utf-8") as f:
                    f.write(text)
            except Exception:
                pass
            self.q.put(("transcript", text))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def open_report(self):
        path = self.dashboard_path()
        if not os.path.exists(path):
            messagebox.showinfo("No dashboard yet",
                                "Run an analysis first — then this opens your dashboard "
                                "with all recordings and your progress.")
            return
        webbrowser.open("file://" + os.path.abspath(path))

    def regenerate(self):
        """Rebuild the dashboard from every saved .result.json — no Azure,
        no Whisper, no Claude, no API usage or cost."""
        library = ec.library_dir()
        has_results = False
        if os.path.isdir(library):
            for _root, _d, files in os.walk(library):
                if any(f.endswith(".result.json") for f in files):
                    has_results = True
                    break
        if not has_results:
            messagebox.showinfo(
                "Nothing to re-build",
                "No saved analyses found yet.\n\nRun Analyze once — after that, "
                "Re-build rebuilds the whole dashboard instantly from saved results, "
                "without calling Azure again.")
            return
        try:
            path = self.dashboard_path()  # project root
            with open(path, "w", encoding="utf-8") as f:
                f.write(ec.build_dashboard_for_dir(library))
            self.status.set("Re-built dashboard (no API calls) — %s" % path)
            webbrowser.open("file://" + os.path.abspath(path))
        except Exception:
            messagebox.showerror("Re-build failed", traceback.format_exc()[-1500:])

    def pick_json(self):
        p = filedialog.askopenfilename(title="Choose grammar analysis JSON",
                                       filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if p:
            self.json_var.set(p)

    def pick_out(self):
        p = filedialog.asksaveasfilename(title="Save report as", defaultextension=".html",
                                         initialfile="report.html",
                                         filetypes=[("HTML", "*.html")])
        if p:
            self.out_var.set(p)

    # ---- run ----
    def run(self):
        if self.busy:
            return
        audio = self.audio_var.get().strip()
        ref = self.transcript.get("1.0", "end").strip() or None
        json_path = self.json_var.get().strip()
        use_json = bool(json_path)
        do_llm = self.do_llm.get() and not use_json
        do_azure = self.do_azure.get()
        out = self.out_var.get().strip()
        strictness = self.strict_labels.get(self.strict_var.get(), "strict")

        if not audio or not os.path.exists(audio):
            messagebox.showerror("Missing audio", "Please choose a valid audio file.")
            return
        if use_json and not os.path.exists(json_path):
            messagebox.showerror("JSON not found", "The grammar analysis JSON file doesn't exist.")
            return
        if not do_llm and not do_azure and not use_json:
            messagebox.showerror("Nothing to do", "Select at least one analysis.")
            return
        if do_azure and not ref:
            messagebox.showerror("Transcript needed",
                                 "Azure pronunciation scoring needs the transcript you read.")
            return
        if do_llm and not self.anthropic_key.get().strip():
            messagebox.showerror("Anthropic key needed",
                                 "Enter an Anthropic API key, untick Whisper+Claude, "
                                 "or load a grammar analysis JSON from Claude.")
            return

        # push keys into env for the worker
        if self.anthropic_key.get().strip():
            os.environ["ANTHROPIC_API_KEY"] = self.anthropic_key.get().strip()
        if self.azure_key.get().strip():
            os.environ["AZURE_SPEECH_KEY"] = self.azure_key.get().strip()
        if self.azure_region.get().strip():
            os.environ["AZURE_SPEECH_REGION"] = self.azure_region.get().strip()

        # remember keys so they only need pasting once
        cfg = load_config()
        cfg.update({
            "anthropic_key": self.anthropic_key.get().strip(),
            "azure_key": self.azure_key.get().strip(),
            "azure_region": self.azure_region.get().strip(),
            "strictness_label": self.strict_var.get(),
        })
        save_config(cfg)

        self.busy = True
        self.run_btn.config(state="disabled")
        self.bar.start(12)
        self.status.set("Starting…")
        threading.Thread(
            target=self.worker,
            args=(audio, ref, do_llm, do_azure, out, json_path, strictness),
            daemon=True,
        ).start()

    def worker(self, audio, ref, do_llm, do_azure, out, json_path, strictness="strict"):
        try:
            base = None
            if json_path:
                with open(json_path, encoding="utf-8") as f:
                    base = json.load(f)
            # each recording gets its own subfolder: VideoAudioFiles/<stem>/
            stem = os.path.splitext(os.path.basename(audio))[0] if audio else "recording"
            library = ec.library_dir()
            rec = ec.rec_dir_for(stem, library=library)
            # make sure the audio (and its txt/json) live in that subfolder
            dest_audio = os.path.join(rec, os.path.basename(audio))
            if os.path.abspath(audio) != os.path.abspath(dest_audio):
                try:
                    shutil.copy2(audio, dest_audio)
                except Exception:
                    dest_audio = audio
            if ref:
                with open(os.path.join(rec, stem + ".txt"), "w", encoding="utf-8") as f:
                    f.write(ref)
            if base is not None:
                with open(os.path.join(rec, stem + ".json"), "w", encoding="utf-8") as f:
                    json.dump(base, f, ensure_ascii=False, indent=2)

            data = ec.analyze_recording(
                dest_audio, reference_text=ref, do_llm=do_llm, do_azure=do_azure,
                base_data=base, strictness=strictness,
                progress=lambda m: self.q.put(("status", m)),
            )
            data.setdefault("title", stem)
            # save the full analysis (drives the dashboard + lets us re-build later
            # WITHOUT re-calling Azure/Whisper/Claude)
            with open(os.path.join(rec, stem + ".result.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if data.get("polished"):
                with open(os.path.join(rec, stem + ".polished.txt"), "w", encoding="utf-8") as f:
                    f.write(data["polished"])
            # log this session for the Summary curve (history lives at the library root)
            ec.log_session(data, os.path.join(library, "history.json"))
            # (re)build the ONE dashboard from the whole library, written to the
            # project root so it's easy to find (not buried with the audio).
            dash = self.dashboard_path()
            with open(dash, "w", encoding="utf-8") as f:
                f.write(ec.build_dashboard_for_dir(library))
            self.q.put(("done", dash))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    # ---- poll worker messages on the UI thread ----
    def poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind == "status":
                    self.status.set(payload)
                elif kind == "transcript":
                    self.finish_transcript(payload)
                elif kind == "done":
                    self.finish_ok(payload)
                elif kind == "error":
                    self.finish_err(payload)
        except queue.Empty:
            pass
        self.root.after(150, self.poll)

    def finish_transcript(self, text):
        self.busy = False
        self.bar.stop()
        self.run_btn.config(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.insert("1.0", text)
        self.status.set("Transcript ready — review/fix it, then Analyze.")

    def finish_ok(self, out):
        self.busy = False
        self.bar.stop()
        self.run_btn.config(state="normal")
        self.status.set("Done — report saved to %s" % out)
        try:
            webbrowser.open("file://" + os.path.abspath(out))
        except Exception:
            pass

    def finish_err(self, tb):
        self.busy = False
        self.bar.stop()
        self.run_btn.config(state="normal")
        self.status.set("Failed — see error dialog.")
        messagebox.showerror("Analysis failed", tb[-1500:])


def main():
    root = tk.Tk()
    CoachGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
