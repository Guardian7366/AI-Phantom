# ai_phantom/core/logger.py
from __future__ import annotations

import csv
import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class RunLogger:
    run_dir: str
    jsonl_path: str
    csv_path: str
    _csv_file: Any
    _jsonl_file: Any
    _csv_writer: Optional[csv.DictWriter] = None
    _csv_fields: Optional[list[str]] = None

    @staticmethod
    def create(base_dir: str = "results/runs", run_name: Optional[str] = None) -> "RunLogger":
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = run_name or f"run_{ts}"
        run_dir = os.path.join(base_dir, name)
        os.makedirs(run_dir, exist_ok=True)

        jsonl_path = os.path.join(run_dir, "metrics.jsonl")
        csv_path = os.path.join(run_dir, "metrics.csv")

        jsonl_f = open(jsonl_path, "a", encoding="utf-8")
        csv_f = open(csv_path, "a", newline="", encoding="utf-8")

        return RunLogger(
            run_dir=run_dir,
            jsonl_path=jsonl_path,
            csv_path=csv_path,
            _csv_file=csv_f,
            _jsonl_file=jsonl_f,
        )

    def log(self, row: Dict[str, Any]) -> None:
        """
        Escribe 1 evento. Seguro: si falla, no rompe el training.
        Formato:
          - JSONL: 1 dict por línea (full fidelity)
          - CSV: columnas estables (se fijan con el primer row)
        """
        try:
            # JSONL
            self._jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
            self._jsonl_file.flush()

            # CSV (fields fijas)
            if self._csv_writer is None:
                self._csv_fields = sorted(list(row.keys()))
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fields)
                if self._csv_file.tell() == 0:
                    self._csv_writer.writeheader()

            out = {}
            for k in self._csv_fields or []:
                v = row.get(k, "")
                # CSV friendly
                if isinstance(v, (dict, list, tuple)):
                    v = json.dumps(v, ensure_ascii=False)
                out[k] = v

            self._csv_writer.writerow(out)
            self._csv_file.flush()
        except Exception as e:
            # no matar entrenamiento por logging
            print(f"⚠️ Logger error (ignored): {e}")

    def close(self) -> None:
        try:
            self._jsonl_file.close()
        except Exception:
            pass
        try:
            self._csv_file.close()
        except Exception:
            pass