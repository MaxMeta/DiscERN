"""Loading a directory of custom reference BGCs (``-r <directory>``)."""

import glob
import os

from .funcs_2 import make_sparse_dict, parse_cb_outputs
from .polymer_tools import make_reference_polymer_dict


class CustomReferenceError(ValueError):
    """A custom reference directory cannot be used as given."""


def file_stem(path):
    """Return the file name without its last extension ('d/x.region001.gbk' -> 'x.region001')."""
    return os.path.basename(path).rsplit(".", 1)[0]


def _key_by_stem(vectors):
    """Re-key a {file_path: vector} dict by file stem."""
    return {file_stem(path): vec for path, vec in (vectors or {}).items()}


def load_custom_references(ref_dir, db_dir, num_cpus=0, poly_search=False,
                           as6=False, mibig_exclude=None):
    """
    Build reference vectors from a directory of custom reference BGCs.

    Returns:
        (labels, gbk_paths, bs_vecs, cb_vecs, pol_vecs), where labels are the
        .gbk file stems and every vector dict is keyed by those stems.

    Raises:
        CustomReferenceError: if the directory cannot be used.
    """
    gbk_paths = sorted(glob.glob(os.path.join(ref_dir, "*.gbk")))
    txt_paths = glob.glob(os.path.join(ref_dir, "*.txt"))
    json_paths = glob.glob(os.path.join(ref_dir, "*.json"))

    if not gbk_paths:
        raise CustomReferenceError(
            f"no .gbk files found in the custom reference directory '{ref_dir}'."
        )

    if len(gbk_paths) != len(txt_paths):
        raise CustomReferenceError(
            "number of reference knownclusterblast text files and gbk files not equal"
        )

    if poly_search and len(gbk_paths) != len(json_paths):
        raise CustomReferenceError("number of json text files and gbk files not equal")

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
    )
    # One flat k-mer token set per reference, like mibig_pol_dict.pkl. Only
    # needed (and .json files only required) with --poly_search.
    pol_vecs = make_reference_polymer_dict(ref_dir, as6=as6) if poly_search else {}

    if mibig_exclude and mibig_exclude != 'self':
        # Drop excluded MIBiG BGCs from the reference blast-vecs.
        for vec in cb_vecs.values():
            for bgc in mibig_exclude:
                vec.pop(bgc, None)

    return labels, gbk_paths, bs_vecs, cb_vecs, pol_vecs
