"""
MIBiG reference protein sets for the diamond blast source (--cb_source diamond).

Two sets can be built:

  4.0          MIBiG 4.0 as published. The default.
  4.0_legacy   MIBiG 4.0 plus the entries that earlier releases (1.4-3.1)
               contained but 4.0 dropped. Each dropped entry is taken from the
               most recent release that still had it. The dropped entries and
               the reasons MIBiG gives for retiring them are listed in
               docs/mibig_dropped_entries.md.

Building a set downloads MIBiG's protein FASTA files (md5-checked), merges
them, builds a DIAMOND database and searches every MIBiG BGC against it to
make the blast-vec/BLAST-rank background. Everything is written to
<db_root>/<set>/, by default discern/mibig_db/ next to the BiG-SLiCE models:

    discern-setup-mibig                    # MIBiG 4.0
    discern-setup-mibig --set 4.0_legacy   # MIBiG 4.0 + dropped entries
"""

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import tempfile
import urllib.request
from collections import OrderedDict
from pathlib import Path

from . import homology

MIBIG_DOWNLOAD = "https://dl.secondarymetabolites.org/mibig"

# release -> (protein FASTA url, md5)
RELEASE_FASTAS = OrderedDict([
    ("4.0", (f"{MIBIG_DOWNLOAD}/mibig_prot_seqs_4.0.fasta", "1dd73356e63d6ba3507f9199cfd22e1c")),
    ("3.1", (f"{MIBIG_DOWNLOAD}/mibig_prot_seqs_3.1.fasta", "b3cacde3f5d6b8ac27d319ce3d2677c1")),
    ("3.0", (f"{MIBIG_DOWNLOAD}/mibig_prot_seqs_3.0.fasta", "81dc4c0c45296ae22709ebea0c19202f")),
    ("2.0", (f"{MIBIG_DOWNLOAD}/mibig_prot_seqs_2.0.fasta", "c8626c785d25ac22ea5d2d447770e883")),
    ("1.4", (f"{MIBIG_DOWNLOAD}/mibig_prot_seqs_1.4.fasta", "118f2c11edde36c81bdea030a0228492")),
])

# set name -> releases, highest priority first
MIBIG_SETS = OrderedDict([
    ("4.0", ("4.0",)),
    ("4.0_legacy", ("4.0", "3.1", "3.0", "2.0", "1.4")),
])

DEFAULT_SET = "4.0"
DEFAULT_DB_ROOT = Path(__file__).resolve().parent / "mibig_db"
DB_ROOT_ENV = "DISCERN_MIBIG_DB"

_VERSION_SUFFIX = re.compile(r"\.\d+$")


class MibigSetError(RuntimeError):
    """A MIBiG set is missing or cannot be built."""


def default_db_root():
    return Path(os.environ.get(DB_ROOT_ENV, DEFAULT_DB_ROOT))


def unversioned(accession):
    """'BGC0000001.5' -> 'BGC0000001'."""
    return _VERSION_SUFFIX.sub("", accession)


# --- release FASTA files ---------------------------------------------------------

def md5sum(path):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch_release_fasta(release, download_dir):
    """Download a release's protein FASTA (if needed) and check its md5."""
    url, md5 = RELEASE_FASTAS[release]
    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    path = download_dir / os.path.basename(url)
    if path.exists() and md5sum(path) == md5:
        return path
    print(f"downloading MIBiG {release} proteins from {url}")
    with tempfile.NamedTemporaryFile(dir=download_dir, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        urllib.request.urlretrieve(url, tmp_path)
        found = md5sum(tmp_path)
        if found != md5:
            raise MibigSetError(f"md5 mismatch for {url}: expected {md5}, got {found}")
        tmp_path.replace(path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return path


def read_release_fasta(path):
    """{unversioned BGC ID: [protein sequences in file order]} from a MIBiG protein FASTA."""
    proteins = OrderedDict()
    current = None
    chunks = []

    def flush():
        if current is not None:
            seq = "".join(chunks).strip()
            if seq:
                proteins.setdefault(current, []).append(seq)

    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                flush()
                current = unversioned(line[1:].split("|", 1)[0])
                chunks = []
            elif line:
                chunks.append(line)
    flush()
    return proteins


def merge_releases(release_proteins):
    """
    Merge releases, highest priority first.

    Args:
        release_proteins: list of (release, {bgc: [seqs]})
    Returns:
        (OrderedDict {bgc: [seqs]} sorted by ID, {bgc: release it came from})
    """
    merged = {}
    origin = {}
    for release, proteins in release_proteins:
        for bgc, seqs in proteins.items():
            if bgc not in merged:
                merged[bgc] = seqs
                origin[bgc] = release
    ordered = OrderedDict((bgc, merged[bgc]) for bgc in sorted(merged))
    return ordered, {bgc: origin[bgc] for bgc in ordered}


# --- building and loading sets -----------------------------------------------------

class MibigSet:
    """A built MIBiG set: DIAMOND database, protein sizes and background hits."""

    def __init__(self, path):
        self.path = Path(path)
        self.name = self.path.name
        manifest_path = self.path / "manifest.json"
        if not manifest_path.exists():
            raise MibigSetError(
                f"MIBiG set '{self.name}' has not been built in {self.path.parent}. "
                f"Run: discern-setup-mibig --set {self.name}"
            )
        with open(manifest_path) as handle:
            self.manifest = json.load(handle)
        with open(self.path / "sizes.json") as handle:
            self.sizes = json.load(handle)
        self.origin = self.manifest["origin"]
        self.fasta = str(self.path / "proteins.fasta")
        self.db = str(self.path / "proteins.dmnd")
        self._background = None

    @property
    def ids(self):
        return set(self.sizes)

    def legacy_ids(self):
        """IDs that are not part of MIBiG 4.0 (taken from an older release)."""
        return {bgc for bgc, release in self.origin.items() if release != "4.0"}

    def background_raw(self):
        """{mibig row: {mibig subject: [score, n_query_genes, n_subject_genes]}}"""
        if self._background is None:
            self._background, _ = homology.load_raw(str(self.path / "background_raw.json.gz"))
        return self._background

    def proteins_by_bgc(self):
        return read_release_fasta(self.fasta)


def load_set(name=DEFAULT_SET, root=None):
    if name not in MIBIG_SETS:
        raise MibigSetError(f"unknown MIBiG set '{name}'; choose from {', '.join(MIBIG_SETS)}")
    return MibigSet(Path(root or default_db_root()) / name)


def build_set(name=DEFAULT_SET, root=None, threads=0, download_dir=None, force=False,
              settings=homology.DEFAULT_SETTINGS, diamond=None):
    """Download, merge, index and search a MIBiG set. Returns the MibigSet."""
    if name not in MIBIG_SETS:
        raise MibigSetError(f"unknown MIBiG set '{name}'; choose from {', '.join(MIBIG_SETS)}")
    root = Path(root or default_db_root())
    out = root / name
    if (out / "manifest.json").exists() and not force:
        print(f"MIBiG set '{name}' already built in {out} (use --force to rebuild)")
        return MibigSet(out)

    threads = threads or os.cpu_count() or 1
    diamond = homology.find_diamond(diamond)
    download_dir = Path(download_dir or root / "downloads")

    releases = MIBIG_SETS[name]
    release_proteins = []
    for release in releases:
        path = fetch_release_fasta(release, download_dir)
        release_proteins.append((release, read_release_fasta(path)))
    merged, origin = merge_releases(release_proteins)

    return build_set_from_proteins(name, merged, origin, out, threads=threads, settings=settings,
                                   diamond=diamond, releases=releases)


def build_set_from_proteins(name, proteins, origin, out, threads=0,
                            settings=homology.DEFAULT_SETTINGS, diamond=None, releases=()):
    """
    Build a set directory from {bgc: [protein sequences]}: FASTA, DIAMOND
    database and the all-against-all background. Returns the MibigSet.
    """
    out = Path(out)
    threads = threads or os.cpu_count() or 1
    diamond = homology.find_diamond(diamond)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    print(f"writing {len(proteins)} BGCs ({sum(map(len, proteins.values()))} proteins)")
    sizes = homology.write_bgc_fasta(proteins.items(), str(out / "proteins.fasta"))
    with open(out / "sizes.json", "w") as handle:
        json.dump(sizes, handle)
    homology.make_diamond_db(str(out / "proteins.fasta"), str(out / "proteins"),
                             threads=threads, diamond=diamond)

    print("searching MIBiG against itself for the blast-vec background")
    work = tempfile.mkdtemp(dir=out)
    try:
        raw = homology.search(str(out / "proteins.fasta"), str(out / "proteins.dmnd"),
                              os.path.join(work, "background"), threads=threads,
                              settings=settings, diamond=diamond)
    finally:
        shutil.rmtree(work, ignore_errors=True)

    meta = {
        "set": name,
        "releases": {r: {"url": RELEASE_FASTAS[r][0], "md5": RELEASE_FASTAS[r][1]} for r in releases},
        "diamond_version": homology.diamond_version(diamond),
        "settings": settings.as_dict(),
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    homology.save_raw(raw, str(out / "background_raw.json.gz"), meta)
    manifest = dict(meta, n_bgcs=len(sizes), n_proteins=sum(sizes.values()), origin=origin)
    with open(out / "manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=1)
    print(f"MIBiG set '{name}' built in {out}")
    return MibigSet(out)


def setup_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download MIBiG protein sequences and build the DIAMOND database and "
                    "background used by 'discern --cb_source diamond'."
    )
    parser.add_argument("--set", default=DEFAULT_SET, choices=list(MIBIG_SETS),
                        help="MIBiG 4.0 as published (default), or 4.0 plus the entries "
                             "dropped since earlier releases (4.0_legacy)")
    parser.add_argument("--db_dir", default=None,
                        help=f"where to build the set (default: ${DB_ROOT_ENV} or {DEFAULT_DB_ROOT})")
    parser.add_argument("--download_dir", default=None,
                        help="where to keep the downloaded MIBiG FASTA files (default: <db_dir>/downloads)")
    parser.add_argument("-c", "--num_cpus", type=int, default=0, help="threads for DIAMOND (default: all)")
    parser.add_argument("--force", action="store_true", help="rebuild even if the set exists")
    args = parser.parse_args(argv)
    try:
        build_set(args.set, root=args.db_dir, threads=args.num_cpus,
                  download_dir=args.download_dir, force=args.force)
    except (MibigSetError, homology.DiamondError) as e:
        raise SystemExit(f"Error: {e}")


if __name__ == "__main__":
    setup_main()
