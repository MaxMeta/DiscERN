from discern.funcs_2 import parse_bgc_file

HEADER = "ClusterBlast scores for record\n\nTable of genes, locations, strands and annotations of query cluster:\n"


def chunk(name):
    return (
        ">>\n"
        f"1. {name}\n"
        "Source: example\n"
        "Type: NRPS\n"
        "Number of proteins with BLAST hits to this cluster: 2\n"
        "Cumulative BLAST score: 300\n\n"
        "Table of genes, locations, strands and annotations of subject cluster:\n\n"
        "Table of Blast hits (query gene, subject gene, %identity, blast score, %coverage, e-value):\n"
        "q1\ts1\t90\t100\t100.0\t1e-50\n"
        "q2\ts2\t90\t200\t100.0\t1e-50\n"
        "q3\ts3\t90\t300\t100.0\t1e-50\n\n"
    )


def test_mibig_version_suffix_is_dropped(tmp_path):
    # antiSMASH 8 writes "BGC0000315.5"; antiSMASH 7 and the bundled MIBiG data use "BGC0000315".
    cb_file = tmp_path / "x_c1.txt"
    cb_file.write_text(HEADER + chunk("BGC0000315.5") + chunk("BGC0001370"))
    assert set(parse_bgc_file(str(cb_file))) == {"BGC0000315", "BGC0001370"}
