"""Loading and checking a directory of custom reference BGCs (``-r <directory>``)."""

import glob
import os

from .funcs_2 import make_sparse_dict, parse_cb_outputs
from .polymer_tools import make_reference_polymer_dict


REFERENCE_FILES_HELP = (
    "Each custom reference needs <name>.gbk (the antiSMASH region GenBank file), "
    "<name>.txt (its KnownClusterBlast output) and, with --poly_search, <name>.json "
    "(the antiSMASH JSON), all sharing the same <name>."
)

REFERENCE_FILES_HELP_DIAMOND = (
    "With --cb_source diamond, each custom reference needs <name>.gbk (the antiSMASH "
    "region GenBank file) and, with --poly_search, <name>.json (the antiSMASH JSON), "
    "sharing the same <name>; KnownClusterBlast .txt files are not used."
)


def _files_help(require_txt):
    return REFERENCE_FILES_HELP if require_txt else REFERENCE_FILES_HELP_DIAMOND


class CustomReferenceError(ValueError):
    """A custom reference directory cannot be used as given."""


def max_achievable_k(custom_refs, poly_search, cb_source="antismash"):
    """
    The highest algorithm support (k) a hit can reach in this run:
    pfam-vec and blast-vec, plus BLAST-rank, plus the polymer method with
    --poly_search. With --cb_source antismash, BLAST-rank cannot run when
    there are custom references, because they never appear in anyone's
    KnownClusterBlast results; with --cb_source diamond they are subjects of
    the search, so BLAST-rank runs for MIBiG, custom and mixed references.
    """
    blast_rank = cb_source == "diamond" or not custom_refs
    return 2 + (1 if blast_rank else 0) + (1 if poly_search else 0)


def cap_min_k(min_k, custom_refs, poly_search, cb_source="antismash"):
    """Return (min_k to use for the trees, max achievable k)."""
    max_k = max_achievable_k(custom_refs, poly_search, cb_source)
    return min(min_k, max_k), max_k


def custom_mode_notice(poly_search):
    """Text printed at the start of a custom-reference run."""
    max_k = max_achievable_k(True, poly_search)
    methods = "pfam-vec, blast-vec, polymer" if poly_search else "pfam-vec, blast-vec"
    return (
        "Custom reference mode:\n"
        "  - BLAST-rank (ol_hits) is unavailable: custom references never appear in each "
        "other's KnownClusterBlast results.\n"
        f"  - The maximum algorithm support is therefore k={max_k} ({methods}), against "
        f"k={max_achievable_k(False, poly_search)} with MIBiG references.\n"
        "  - The README's k>=3 precision figure was benchmarked with MIBiG references "
        "and does not apply here."
    )


def file_stem(path):
    """Return the file name without its last extension ('d/x.region001.gbk' -> 'x.region001')."""
    return os.path.basename(path).rsplit(".", 1)[0]


def _key_by_stem(vectors):
    """Re-key a {file_path: vector} dict by file stem."""
    return {file_stem(path): vec for path, vec in (vectors or {}).items()}


def _unmatched(labels, names, ext):
    """Describe .gbk stems without a matching file of type ext, and vice versa."""
    labels, names = set(labels), set(names)
    problems = []
    missing = sorted(labels - names)
    extra = sorted(names - labels)
    if missing:
        problems.append(f"no {ext} for {', '.join(missing)}")
    if extra:
        problems.append(f"no .gbk for {', '.join(n + ext for n in extra)}")
    return problems


def _raise_unmatched(ref_dir, problems, require_txt=True):
    raise CustomReferenceError(
        f"the files in the custom reference directory '{ref_dir}' do not match up. "
        f"{_files_help(require_txt)} Unmatched: {'; '.join(problems)}."
    )


def check_reference_files(ref_dir, poly_search=False, require_txt=True):
    """
    Check that every reference has matching .gbk / .txt (/ .json) files.
    require_txt is False for --cb_source diamond, which does not use
    KnownClusterBlast output; names must then be usable as FASTA IDs.

    Returns:
        The sorted list of reference .gbk paths.

    Raises:
        CustomReferenceError: if there are no .gbk files or the names do not match.
    """
    gbk_paths = sorted(glob.glob(os.path.join(ref_dir, "*.gbk")))
    if not gbk_paths:
        raise CustomReferenceError(
            f"no .gbk files found in the custom reference directory '{ref_dir}'. "
            f"{_files_help(require_txt)}"
        )
    labels = [file_stem(path) for path in gbk_paths]

    if not require_txt:
        bad = [label for label in labels if "|" in label or any(c.isspace() for c in label)]
        if bad:
            raise CustomReferenceError(
                "custom reference names must not contain '|' or whitespace: "
                + ", ".join(f"{label}.gbk" for label in bad)
            )

    extensions = [".txt"] if require_txt else []
    if poly_search:
        extensions.append(".json")
    problems = []
    for ext in extensions:
        names = [file_stem(path) for path in glob.glob(os.path.join(ref_dir, "*" + ext))]
        problems.extend(_unmatched(labels, names, ext))
    if problems:
        _raise_unmatched(ref_dir, problems, require_txt)

    return gbk_paths


def validate_reference_vectors(ref_dir, labels, bs_vecs, cb_vecs, pol_vecs=None,
                               poly_search=False):
    """
    Check that every reference label has a vector for each method in use, so
    that later lookups by label cannot fail with a KeyError. cb_vecs is None
    with --cb_source diamond, where blast vectors are built later.

    Raises:
        CustomReferenceError: if a vector is missing.
    """
    problems = _unmatched(labels, cb_vecs, ".txt") if cb_vecs is not None else []
    if poly_search:
        problems.extend(_unmatched(labels, pol_vecs or {}, ".json"))
    if problems:
        _raise_unmatched(ref_dir, problems, require_txt=cb_vecs is not None)

    # dataframe_to_sparse_dict drops BGCs with no biosynthetic Pfam hits.
    no_bs = sorted(set(labels) - set(bs_vecs))
    if no_bs:
        raise CustomReferenceError(
            "no biosynthetic Pfam domains were found in "
            f"{', '.join(n + '.gbk' for n in no_bs)}, so there is no pfam-vec vector "
            "for this reference. Check that it is an antiSMASH region GenBank file with "
            "CDS translations, or remove it from the reference directory."
        )

    for label in labels if cb_vecs is not None else []:
        if not cb_vecs[label]:
            print(
                f"warning: no KnownClusterBlast hits were read from {label}.txt. Check that "
                "it is the knownclusterblast/<record>_c<N>.txt file from an antiSMASH run "
                "with --cb-knownclusters; otherwise the blast-vec method gets no information "
                "from this reference."
            )


def load_custom_references(ref_dir, db_dir, num_cpus=0, poly_search=False,
                           as6=False, mibig_exclude=None, cb_source="antismash"):
    """
    Build reference vectors from a directory of custom reference BGCs.

    Returns:
        (labels, gbk_paths, bs_vecs, cb_vecs, pol_vecs), where labels are the
        .gbk file stems and every vector dict is keyed by those stems.
        pol_vecs is empty unless poly_search is set. cb_vecs is None with
        cb_source "diamond": those vectors come from the DIAMOND search
        (cb_sources.diamond_cb_vectors), which needs only the .gbk files.

    Raises:
        CustomReferenceError: if the directory cannot be used.
    """
    use_kcb = cb_source != "diamond"
    gbk_paths = check_reference_files(ref_dir, poly_search, require_txt=use_kcb)
    labels = [file_stem(path) for path in gbk_paths]

    if len(labels) == 1:
        print(
            "warning: only one custom reference BGC was found. The reference centroid is "
            "that BGC itself, so the trained pfam-vec and blast-vec distance thresholds "
            "degenerate to 0 and will match (almost) nothing. Use two or more references, "
            "or set --bigslice_cutoff / --clusterblast_cutoff manually."
        )

    # The loaders below glob inside a directory, so they take the directory path.
    bs_vecs = _key_by_stem(
        make_sparse_dict(ref_dir, db_dir, num_cpus=num_cpus, glob_pattern="*.gbk")
    )
    cb_vecs = _key_by_stem(
        parse_cb_outputs(ref_dir, glob_pattern="*.txt", num_cpus=num_cpus)
    ) if use_kcb else None
    # One flat k-mer token set per reference, like mibig_pol_dict.pkl. Only
    # needed (and .json files only required) with --poly_search.
    pol_vecs = make_reference_polymer_dict(ref_dir, as6=as6) if poly_search else {}

    validate_reference_vectors(ref_dir, labels, bs_vecs, cb_vecs, pol_vecs, poly_search)

    if use_kcb and mibig_exclude and mibig_exclude != 'self':
        # Drop excluded MIBiG BGCs from the reference blast-vecs.
        for vec in cb_vecs.values():
            for bgc in mibig_exclude:
                vec.pop(bgc, None)

    return labels, gbk_paths, bs_vecs, cb_vecs, pol_vecs
