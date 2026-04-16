from __future__ import annotations

import json
from pathlib import Path
import re
import zipfile

import pandas as pd

from src.constants import REQUIRED_FILES
from src.data.schemas import assert_required_files, validate_transactions_schema
from src.types import DatasetPaths


class DatasetLoader:
	def __init__(self, input_path: str):
		self.input_path = input_path
		self._resolved_input = self.resolve_dataset_root()
		self._dataset_root = self.extract_if_zip()

	def resolve_dataset_root(self) -> str:
		given = Path(self.input_path)
		candidates = [
			given,
			Path(str(given).replace(" ", "+")),
			Path(str(given).replace("+", " ")),
			Path("hackTheCode") / given,
			Path("hackTheCode") / Path(str(given).replace(" ", "+")),
			Path("hackTheCode") / Path(str(given).replace("+", " ")),
		]

		for candidate in candidates:
			if candidate.exists():
				return str(candidate)

		basename = given.name
		patterns = {
			basename,
			basename.replace(" ", "+"),
			basename.replace("+", " "),
		}
		for pattern in patterns:
			found = list(Path(".").glob(f"**/{pattern}"))
			if found:
				return str(found[0])

		raise FileNotFoundError(f"Input path does not exist: {self.input_path}")

	def extract_if_zip(self) -> str:
		resolved = Path(self._resolved_input)
		if resolved.is_dir():
			root = self._find_dataset_root(resolved)
			assert_required_files(str(root))
			return str(root)

		if resolved.suffix.lower() != ".zip":
			raise ValueError(f"Input must be a .zip or directory: {resolved}")

		safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", resolved.stem)
		out_dir = Path("cache") / "extracted" / safe_name
		out_dir.mkdir(parents=True, exist_ok=True)

		with zipfile.ZipFile(resolved, "r") as zf:
			for member in zf.infolist():
				if "__MACOSX" in member.filename or member.filename.endswith(".DS_Store"):
					continue
				zf.extract(member, out_dir)

		root = self._find_dataset_root(out_dir)
		assert_required_files(str(root))
		return str(root)

	def _find_dataset_root(self, base: Path) -> Path:
		if self._contains_required_files(base):
			return base

		for path in sorted(base.rglob("*")):
			if not path.is_dir():
				continue
			if "__MACOSX" in path.parts:
				continue
			if self._contains_required_files(path):
				return path

		raise FileNotFoundError(f"Could not locate dataset root with required files under: {base}")

	@staticmethod
	def _contains_required_files(path: Path) -> bool:
		files = {
			p.name
			for p in path.iterdir()
			if p.is_file() and p.name != ".DS_Store" and not p.name.startswith("._")
		}
		return set(REQUIRED_FILES.values()).issubset(files)

	def load_transactions(self) -> pd.DataFrame:
		p = Path(self._dataset_root) / REQUIRED_FILES["transactions"]
		if not p.exists():
			raise FileNotFoundError(f"Missing required file: {p}")
		df = pd.read_csv(p)
		validate_transactions_schema(df)
		return df

	def load_users(self) -> list[dict]:
		return self._load_json_list("users")

	def load_locations(self) -> list[dict]:
		return self._load_json_list("locations")

	def load_sms(self) -> list[dict]:
		return self._load_json_list("sms")

	def load_mails(self) -> list[dict]:
		return self._load_json_list("mails")

	def _load_json_list(self, key: str) -> list[dict]:
		p = Path(self._dataset_root) / REQUIRED_FILES[key]
		if not p.exists():
			raise FileNotFoundError(f"Missing required file: {p}")
		with open(p, "r", encoding="utf-8") as handle:
			payload = json.load(handle)
		if not isinstance(payload, list):
			raise ValueError(f"{p.name} must contain a JSON list")
		return payload

	def load_all(self) -> DatasetPaths:
		root = Path(self._dataset_root)
		dataset_name = root.name.strip().replace(" ", "_").replace("+", "_").lower()
		return DatasetPaths(
			input_path=self.input_path,
			dataset_root=str(root),
			dataset_name=dataset_name,
			transactions_path=str(root / REQUIRED_FILES["transactions"]),
			users_path=str(root / REQUIRED_FILES["users"]),
			locations_path=str(root / REQUIRED_FILES["locations"]),
			sms_path=str(root / REQUIRED_FILES["sms"]),
			mails_path=str(root / REQUIRED_FILES["mails"]),
		)


class PairedDatasetLoader:
	def __init__(self, reference_path: str, target_path: str):
		self.reference_path = reference_path
		self.target_path = target_path

	def load_reference(self) -> DatasetPaths:
		return DatasetLoader(self.reference_path).load_all()

	def load_target(self) -> DatasetPaths:
		return DatasetLoader(self.target_path).load_all()

	def load_both(self) -> tuple[DatasetPaths, DatasetPaths]:
		return self.load_reference(), self.load_target()
